import os
import time
import google.generativeai as genai
from google.generativeai.types import Tool, FunctionDeclaration
from PIL.Image import Image
from sqlalchemy.orm import Session
import traceback

# Internal imports
from database import crud, models as db_models
from chat import models as chat_models
from chat import tools


# =====================
# GEMINI API KEY CONFIGURATION
# =====================

# ⚠️ Recommended: use environment variable, fallback to hardcoded for local dev
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCFJJqzSuBMNmCDhc5-QjJgRH87F1P3M_A")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Set it in your environment or .env file.")

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"❌ Error configuring Gemini: {e}")


# =====================
# TOOL DEFINITIONS
# =====================

run_python_tool_schema = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="run_python_code",
            description="""Executes Python code for data analysis and visualization. 

IMPORTANT: Only use this tool when the user explicitly asks for data analysis, visualization, statistics, or code execution. Do NOT use this tool for general conversational questions, greetings, or questions about your capabilities. 

DATASET: If loaded, available as pandas DataFrame 'df' with columns already accessible.

PRE-IMPORTED LIBRARIES: pandas (pd), matplotlib.pyplot (plt), seaborn (sns), numpy (np), sklearn (all modules), io, base64, json

OUTPUT FORMAT:
- Use print() for all text outputs, statistics, and data summaries
- For DataFrames, Series, or statistics tables: use `print(df.to_markdown())` or `print(statistics.to_markdown())` to create nicely formatted markdown tables
- For visualizations: 
  1. Check column is numeric: if col not in df.select_dtypes(include=[np.number]).columns: skip or convert
  2. Create plot with plt or sns (handle errors with try-except)
  3. Save: buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', dpi=100); buf.seek(0)
  4. Encode: img_base64 = base64.b64encode(buf.read()).decode('utf-8')
  5. Print: print(f"PLOT_IMG_BASE64:{img_base64}")
  6. Close: plt.close()

EXAMPLES:
- Summary with descriptive statistics:
  print("### Missing Values")
  missing = df.isnull().sum()
  missing_df = pd.DataFrame({'Column': missing.index, 'Missing Count': missing.values, 'Percentage': (missing.values / len(df) * 100)})
  print(missing_df.to_markdown())
  
  print("\n### Columns")
  cols_df = pd.DataFrame({
      'Column': df.columns,
      'Type': [str(df[col].dtype) for col in df.columns],
      'Description': ['Description here'] * len(df.columns)
  })
  print(cols_df.to_markdown())
  
  print("\n### Descriptive Statistics")
  desc_stats = df.describe()
  print(desc_stats.to_markdown())
- Visualization: 
  numeric_cols = df.select_dtypes(include=[np.number]).columns
  if len(numeric_cols) > 0:
      plt.figure(figsize=(10, 6))
      sns.histplot(df[numeric_cols[0]], bins=20)
      plt.title(f'Distribution of {numeric_cols[0]}')
      [save steps above]
- Analysis: print(f"Mean: {df['column'].mean()}")  # Only for numeric columns

IMPORTANT: Always use try-except for error handling. Check data types before operations!""",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "code": {"type": "STRING", "description": "Complete Python code to execute. Must use print() for outputs. For plots, save as base64 with PLOT_IMG_BASE64 prefix."}
                },
                "required": ["code"]
            }
        )
    ]
)

google_search_tool_schema = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="google_search",
            description="Simulates a Google Search for educational use (returns mock data).",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING", "description": "Search query string."}
                },
                "required": ["query"]
            }
        )
    ]
)


# =====================
# MAIN AGENT CLASS
# =====================

class DataAnalystAgent:
    """Main DataGem AI Agent that handles Gemini interaction, tool calls, and chat history."""

    def __init__(self, db: Session, user: db_models.User, dataset: list[dict] | None = None):
        self.db = db
        self.user = user
        self.dataset = dataset
        self.model = None
        self.chat = None

        try:
            # Build system instruction - comprehensive and clear
            system_instruction = """You are DataGem, a friendly AI assistant and expert data analyst. You can have natural conversations AND analyze data.

⚠️ CRITICAL: DECIDE FIRST - Is this a conversation or data analysis?

STEP 1: CLASSIFY THE USER'S QUESTION:

A) CONVERSATIONAL (Answer directly, NO tools):
   ✅ Greetings: "hi", "hello", "hey", "how are you?", "what's up"
   ✅ General questions: "what can you do?", "what are your capabilities?", "how does this work?", "what all functions can you perform?"
   ✅ Questions about you: "who are you?", "what is DataGem?", "tell me about yourself"
   ✅ General chat: "thanks", "thank you", "goodbye", "bye", "okay", "cool"
   ✅ Questions that don't mention data, analysis, statistics, charts, or visualizations
   
   → For these: Respond naturally and conversationally. Do NOT use any tools. Just chat!

B) DATA ANALYSIS (Use tools):
   ✅ Explicit requests: "show me statistics", "analyze the data", "what are the correlations?"
   ✅ Visualization requests: "create a chart", "show me a histogram", "plot the data", "visualize"
   ✅ Analysis requests: "find outliers", "what insights can you find?", "analyze data quality"
   ✅ Code/calculation requests: "run analysis", "calculate statistics", "perform analysis"
   ✅ Questions that explicitly ask to analyze, visualize, or process data
   
   → For these: Use the run_python_code tool

STEP 2: CONTEXT AWARENESS:
   - If dataset is loaded: You can mention it in conversational answers (e.g., "I can analyze your dataset with 500 rows and 7 columns")
   - If no dataset: Mention that in conversational answers
   - Always be helpful and guide users on what you can do

DATA ANALYSIS RULES (when using tools):
1. ALWAYS use the run_python_code tool when asked to analyze data or create visualizations
2. The dataset is already loaded as a pandas DataFrame named 'df' - use it directly
3. NEVER ask the user for data - it's already available
4. ALWAYS check data types before operations - use df.select_dtypes(include=[np.number]) for numeric columns
5. ALWAYS use try-except blocks for error handling in your code
6. If code fails, analyze the error and provide a corrected version
7. ALWAYS provide comprehensive text summaries with tables and formatted text after running code
8. For visualizations: ONLY use numeric columns, create plots, save as base64, and explain what they show
9. When generating summaries: ALWAYS include descriptive statistics using df.describe().to_markdown() - this is REQUIRED

AVAILABLE LIBRARIES (pre-imported):
- pandas (pd) - data manipulation
- matplotlib.pyplot (plt) - plotting
- seaborn (sns) - statistical visualizations  
- numpy (np) - numerical operations
- sklearn - machine learning (all modules: LinearRegression, train_test_split, metrics, etc.)
- io, base64, json - for saving/encoding images

WHEN CREATING VISUALIZATIONS:
1. ALWAYS check if columns are numeric before plotting: numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
2. Only use numeric columns for plots (avoid string/ID columns)
3. Handle errors gracefully with try-except blocks
4. Create the plot using matplotlib or seaborn
5. Save to BytesIO buffer: buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
6. Encode as base64: buf.seek(0); img_base64 = base64.b64encode(buf.read()).decode('utf-8')
7. Print with prefix: print(f"PLOT_IMG_BASE64:{img_base64}")
8. Always close the figure: plt.close()

RESPONSE FORMAT (ONLY for data analysis questions):
- For conversational questions: Just respond naturally, no code, no tools
- For data analysis questions:
  - ALWAYS show the code you're running (it will be displayed automatically)
  - Run code using the tool
  - IMPORTANT: When showing DataFrames or statistics, use `print(df.to_markdown())` or `print(statistics.to_markdown())` to create formatted tables
  - After code execution, provide a COMPREHENSIVE TEXT SUMMARY that includes:
    * Clear explanations in markdown format
    * Tables showing key statistics, correlations, or data summaries (use markdown table format)
    * Bullet points for insights
    * Headings and proper formatting
    * Interpretation of results
    * Actionable recommendations

TEXT SUMMARY REQUIREMENTS (CRITICAL - Format like ChatGPT):
- ALWAYS write comprehensive text summaries after code execution - this is MANDATORY
- Format EXACTLY like ChatGPT with this structure:
  * Start: "Here's the detailed **summary** of your dataset [name]:"
  * Section 1: "### 📊 Dataset Overview" with bullet points (Total Records, Total Columns, missing values)
  * Section 2: "### 🧩 Columns" with markdown table (Column | Type | Description)
  * Section 3: "### 📈 Descriptive Statistics" with markdown table (Feature | Mean | Std Dev | Min | 25% | 50% | 75% | Max)
  * End: "Would you like me to:" with numbered options
- Use emojis in ALL section headings (📊, 🧩, 📈)
- Include checkmarks (✅) for positive indicators
- Use proper markdown table syntax with alignment
- Extract REAL values from code output - never use placeholders like [number]
- Write in friendly, conversational ChatGPT-style tone
- ALWAYS generate this summary - it's the most important part of your response
- When the user asks for a summary, generate code that prints tables using to_markdown(), then provide a text explanation

VISUALIZATION REQUIREMENTS:
- ALWAYS create visualizations when asked
- Use clear, informative titles and labels
- Make plots readable and professional
- Save ALL plots as base64 images

CODE REQUIREMENTS:
- Show complete, runnable code
- Include comments for clarity
- Handle edge cases
- Use appropriate libraries for the task

Be helpful, thorough, and always provide value with comprehensive summaries including tables!"""
            
            # Initialize the Gemini model with tools and function calling config
            self.model = genai.GenerativeModel(
                model_name="models/gemini-2.5-flash",  # Use flash or pro depending on speed/quality preference
                tools=[run_python_tool_schema, google_search_tool_schema],
                tool_config={
                    "function_calling_config": {
                        "mode": "ANY",  # ANY enables constrained decoding for better function call handling (prevents invalid function calls)
                    }
                },
                system_instruction=system_instruction
            )
            
            # Create a separate model WITHOUT tools for generating text summaries
            self.text_model = genai.GenerativeModel(
                model_name="models/gemini-2.5-flash",
                system_instruction="You are DataGem, an expert data analyst. You provide comprehensive text summaries with formatted markdown tables and insights. Always write clear, helpful explanations."
            )

            # Load chat history from DB
            history_db = crud.get_chat_history(db=self.db, user_id=self.user.id)
            history_gemini = self.convert_db_history_to_gemini(history_db)

            # Start a Gemini chat session
            self.chat = self.model.start_chat(history=history_gemini)
            # Removed "Ready" message - it might interfere with responses

        except Exception as e:
            print(f"❌ Error initializing DataAnalystAgent: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    def convert_db_history_to_gemini(self, history_db: list[db_models.ChatHistory]) -> list[dict]:
        """Convert stored SQL chat history into Gemini-compatible format."""
        gemini_history = []
        for msg in reversed(history_db):
            gemini_history.append({
                "role": msg.role,
                "parts": [{"text": msg.content}]
            })
        return gemini_history

    # ------------------------------------------------------------------
    async def stream_response(self, prompt: str, image: Image | None = None, max_iterations: int = 10):
        """Streams Gemini's response, handles tool calls, and saves messages to DB."""
        
        print(f"📨 Processing prompt: {prompt[:100]}...")

        crud.save_chat_message(
            db=self.db,
            user_id=self.user.id,
            role="user",
            content=prompt
        )

        # Enhance prompt with dataset context if available (simplified)
        enhanced_prompt = prompt
        if self.dataset and len(self.dataset) > 0:
            # Get column names and sample info
            sample_row = self.dataset[0] if self.dataset else {}
            columns = list(sample_row.keys()) if sample_row else []
            row_count = len(self.dataset)
            
            # Enhanced context with clear instructions
            # Detect numeric columns more reliably
            numeric_cols = []
            if self.dataset and len(self.dataset) > 0:
                for col in columns:
                    try:
                        # Check first few rows to see if values are numeric
                        sample_values = [self.dataset[i].get(col, '') for i in range(min(5, len(self.dataset)))]
                        numeric_count = sum(1 for v in sample_values if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit()))
                        if numeric_count >= 2:  # At least 2 numeric values
                            numeric_cols.append(col)
                    except:
                        pass
            
            dataset_context = f"""User question: "{prompt}"

⚠️ FIRST: Classify this question - Is it conversational or data analysis?

Dataset available (for reference):
- {row_count} rows, {len(columns)} columns
- All columns: {', '.join(columns)}
- Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None detected'}
- DataFrame name: 'df'

DECISION LOGIC:

A) If CONVERSATIONAL (greetings like "hi"/"hey", "what can you do?", general questions):
   → Answer directly WITHOUT using any tools
   → You can mention: "I can analyze your dataset with {row_count} rows and {len(columns)} columns"
   → Be friendly and conversational
   → Do NOT call run_python_code tool

B) If DATA ANALYSIS (mentions: statistics, analyze, visualize, chart, plot, insights, correlations):
   → Use run_python_code tool to analyze or visualize
   → For visualizations, save plots as base64 images using PLOT_IMG_BASE64 format
   → After running code, provide comprehensive text summary with tables and insights

Remember: For simple greetings or general questions, just chat naturally - no tools needed!"""
            enhanced_prompt = dataset_context
            print(f"📊 Enhanced prompt with dataset context ({row_count} rows, {len(columns)} columns)")

        prompt_parts = [enhanced_prompt]
        if image:
            prompt_parts.append(image)

        ai_response_content = ""
        has_output = False
        iteration_count = 0

        try:
            # Stream Gemini output
            print("🔄 Starting Gemini stream...")
            try:
                response_stream = self.chat.send_message(prompt_parts, stream=True)
                print("✅ Stream object created, starting to iterate...")
            except Exception as stream_init_error:
                print(f"❌ Error creating stream: {stream_init_error}")
                yield f"❌ Error initializing response stream: {str(stream_init_error)}"
                return

            event_count = 0
            last_event_time = time.time()
            STREAM_TIMEOUT = 30  # 30 second timeout for stream
            
            try:
                for event in response_stream:
                    event_count += 1
                    current_time = time.time()
                    
                    # Check for timeout
                    if current_time - last_event_time > STREAM_TIMEOUT:
                        print(f"⏱️ Stream timeout: No events for {STREAM_TIMEOUT} seconds")
                        yield "\n\n⚠️ Stream timeout: The response is taking too long. Please try again.\n"
                        break
                    
                    last_event_time = current_time
                    
                    if event_count == 1:
                        print(f"✅ First event received (type: {type(event)})")
                    if event_count % 10 == 0:
                        print(f"📊 Processed {event_count} events so far...")
                    
                    iteration_count += 1
                    if iteration_count > max_iterations * 1000:  # Increased safety limit
                        yield "\n⚠️ Maximum iterations reached. Stopping to prevent infinite loop.\n"
                        break
                    
                    # --- Extract function calls from candidates or parts ---
                    function_call_to_process = None
                    function_call_found = False
                    
                    # First, check candidates (most reliable)
                    if hasattr(event, "candidates") and event.candidates:
                        for candidate in event.candidates:
                            # Check finish_reason for invalid function calls
                            if hasattr(candidate, "finish_reason"):
                                finish_reason = candidate.finish_reason
                                if finish_reason == 10:  # FUNCTION_CALL_INVALID
                                    print(f"⚠️ Invalid function call detected (finish_reason: {finish_reason})")
                                    # Skip this candidate and try to extract text if available
                                    if hasattr(candidate, "content") and candidate.content:
                                        if hasattr(candidate.content, "parts"):
                                            for part in candidate.content.parts:
                                                if hasattr(part, "text") and part.text:
                                                    has_output = True
                                                    ai_response_content += part.text
                                                    yield part.text
                                    continue
                            
                            if hasattr(candidate, "content") and candidate.content:
                                if hasattr(candidate.content, "parts"):
                                    for part in candidate.content.parts:
                                        # Check for function calls
                                        if hasattr(part, "function_call") and part.function_call:
                                            fc = part.function_call
                                            tool_name = fc.name if hasattr(fc, "name") else "unknown"
                                            tool_args = dict(fc.args) if hasattr(fc, "args") and fc.args else {}
                                            function_call_to_process = (tool_name, tool_args)
                                            function_call_found = True
                                            break
                                        
                                        # Check for text in parts
                                        if hasattr(part, "text") and part.text:
                                            has_output = True
                                            ai_response_content += part.text
                                            yield part.text
                            
                            if function_call_found:
                                break
                    
                    # If no function call in candidates, check event.parts (fallback)
                    if not function_call_found and hasattr(event, "parts") and event.parts:
                        for part in event.parts:
                            if hasattr(part, "function_call") and part.function_call:
                                fc = part.function_call
                                tool_name = fc.name if hasattr(fc, "name") else "unknown"
                                tool_args = dict(fc.args) if hasattr(fc, "args") and fc.args else {}
                                function_call_to_process = (tool_name, tool_args)
                                function_call_found = True
                                break
                    
                    # --- Process function call if found ---
                    if function_call_found and function_call_to_process:
                        tool_name, tool_args = function_call_to_process
                        print(f"🔧 Executing tool: {tool_name}")
                        yield f"\n\n🤖 **Executing:** `{tool_name}`\n\n"
                        
                        try:
                            # Initialize tool_result to ensure it's always defined
                            tool_result = None
                            
                            if tool_name == "run_python_code":
                                code = tool_args.get("code", "")
                                if not code:
                                    tool_result = "Error: No code provided to execute."
                                else:
                                    # Show the code being run BEFORE execution
                                    yield f"```python\n{code}\n```\n\n"
                                    print(f"💻 Running Python code ({len(code)} chars)...")
                                    tool_result = tools.run_python_code(code, self.dataset)
                                    print(f"✅ Code execution completed")
                                    # Yield the tool result output so it's visible
                                    if tool_result:
                                        yield f"\n**Code Output:**\n```\n{tool_result}\n```\n\n"
                            elif tool_name == "google_search":
                                query = tool_args.get("query", "")
                                tool_result = tools.google_search(query)
                            else:
                                tool_result = f"Error: Unknown tool `{tool_name}`"
                            
                            # Ensure tool_result is set
                            if tool_result is None:
                                tool_result = "Error: Tool execution returned no result."
                            
                            # Feed tool response back to Gemini
                            has_image = "PLOT_IMG_BASE64:" in tool_result if tool_result else False
                            
                            # Check if code execution failed
                            code_failed = "Code execution failed" in tool_result or "Error:" in tool_result if tool_result else False
                            
                            # Extract key information from tool result for better context
                            # Use larger preview to ensure all tables are included
                            tool_result_preview = tool_result[:8000] if tool_result else 'No results'
                            if tool_result and len(tool_result) > 8000:
                                tool_result_preview += '\n... (truncated, showing first 8000 chars)'
                            
                            # Always define followup_prompt
                            if code_failed:
                                followup_prompt = f"""The Python code execution encountered an error. Here are the details:

{tool_result_preview}

IMPORTANT: Even though the code failed, you MUST still provide a comprehensive text summary based on what you know about the dataset. Use the dataset information from the original context.

CRITICAL: You MUST write a comprehensive text summary NOW. Do NOT use any tools - just write text directly in your response.

INSTRUCTIONS:
1. Parse the code output above to extract ALL tables and data
2. Extract actual numeric values, column names, types, and statistics
3. Create markdown tables with proper alignment using the actual extracted values
4. Format your response EXACTLY like this ChatGPT-style example:

Here's the detailed **summary** of your dataset `student_exam_scores.csv`:

### 📊 Dataset Overview

* **Total Records (Rows):** 200
* **Total Columns:** 6
* **No missing values detected** ✅

### 🧩 Columns

| Column               | Type    | Description                          |
| -------------------- | ------- | ------------------------------------ |
| `student_id`         | object  | Unique ID for each student           |
| `hours_studied`      | float64 | Number of study hours per day        |
| `sleep_hours`        | float64 | Average hours of sleep               |
| `attendance_percent` | float64 | Attendance percentage                |
| `previous_scores`    | int64   | Past academic performance score      |
| `exam_score`         | float64 | Current exam score (target variable) |

### 📈 Descriptive Statistics

| Feature                | Mean  | Std Dev | Min  | 25%  | 50%   | 75%   | Max   |
| ---------------------- | ----- | ------- | ---- | ---- | ----- | ----- | ----- |
| **hours_studied**      | 6.33  | 3.23    | 1.0  | 3.5  | 6.15  | 9.0   | 12.0  |
| **sleep_hours**        | 6.62  | 1.50    | 4.0  | 5.3  | 6.7   | 8.03  | 9.0   |
| **attendance_percent** | 74.83 | 14.25   | 50.3 | 62.2 | 75.25 | 87.43 | 100.0 |
| **previous_scores**    | 66.8  | 15.66   | 40   | 54   | 67.5  | 80    | 95    |
| **exam_score**         | 33.96 | 6.79    | 17.1 | 29.5 | 34.05 | 38.75 | 51.3  |

---

Would you like me to:

1. Generate **visual insights** (correlation heatmap, distributions, scatterplots)?
2. Build a **predictive model** (e.g., linear regression to predict `exam_score`)?
3. Create a **summary report (PDF/HTML)** for this dataset?

MANDATORY REQUIREMENTS:
- Start with "Here's the detailed **summary** of your dataset" (include dataset name if available)
- Use EXACT emoji headings: 📊 Dataset Overview, 🧩 Columns, 📈 Descriptive Statistics
- Include bullet points with bold labels (e.g., **Total Records (Rows):**)
- Create markdown tables with proper alignment using ACTUAL values from the code output above
- Use checkmarks (✅) for positive indicators
- End with "Would you like me to:" followed by numbered options
- Write in a friendly, conversational ChatGPT-style tone
- PARSE the code output tables and extract ALL actual values - do NOT use placeholders like [number] or [type]
- Include ALL columns and statistics that appear in the code output
{"- Describe any visualizations that were generated" if has_image else ""}

CRITICAL: The code output above contains tables with actual data. You MUST parse those tables and recreate them as markdown tables in your response with the EXACT same values. Do not summarize - show the complete tables.

IMPORTANT: Start your response with "🤖 DataGem:" and write ONLY text - no code blocks, no function calls, just formatted markdown text with tables and explanations.

Write the summary NOW - do not use tools, just write the text directly."""
                            else:
                                followup_prompt = f"""The Python code has been executed successfully. Here are the results:

{tool_result_preview}

CRITICAL: You MUST write a comprehensive text summary NOW. Do NOT use any tools - just write text directly in your response.

IMPORTANT: The code output above contains tables and data. Parse those tables and extract the actual values. Then write a comprehensive text summary with formatted markdown tables.

INSTRUCTIONS:
1. Parse the code output above to extract ALL tables and data
2. Extract actual numeric values, column names, types, and statistics from the output
3. Create markdown tables with proper alignment using the actual extracted values
4. Format your response EXACTLY like this ChatGPT-style example:

🤖 DataGem: Here's the detailed **summary** of your dataset `student_exam_scores.csv`:

### 📊 Dataset Overview

* **Total Records (Rows):** 200
* **Total Columns:** 6
* **No missing values detected** ✅

### 🧩 Columns

| Column               | Type    | Description                          |
| -------------------- | ------- | ------------------------------------ |
| `student_id`         | object  | Unique ID for each student           |
| `hours_studied`      | float64 | Number of study hours per day        |
| `sleep_hours`        | float64 | Average hours of sleep               |
| `attendance_percent` | float64 | Attendance percentage                |
| `previous_scores`    | int64   | Past academic performance score      |
| `exam_score`         | float64 | Current exam score (target variable) |

### 📈 Descriptive Statistics

| Feature                | Mean  | Std Dev | Min  | 25%  | 50%   | 75%   | Max   |
| ---------------------- | ----- | ------- | ---- | ---- | ----- | ----- | ----- |
| **hours_studied**      | 6.33  | 3.23    | 1.0  | 3.5  | 6.15  | 9.0   | 12.0  |
| **sleep_hours**        | 6.62  | 1.50    | 4.0  | 5.3  | 6.7   | 8.03  | 9.0   |
| **attendance_percent** | 74.83 | 14.25   | 50.3 | 62.2 | 75.25 | 87.43 | 100.0 |
| **previous_scores**    | 66.8  | 15.66   | 40   | 54   | 67.5  | 80    | 95    |
| **exam_score**         | 33.96 | 6.79    | 17.1 | 29.5 | 34.05 | 38.75 | 51.3  |

---

Would you like me to:

1. Generate **visual insights** (correlation heatmap, distributions, scatterplots)?
2. Build a **predictive model** (e.g., linear regression to predict `exam_score`)?
3. Create a **summary report (PDF/HTML)** for this dataset?

MANDATORY REQUIREMENTS:
- Start with "Here's the detailed **summary** of your dataset" (include dataset name if available)
- Use EXACT emoji headings: 📊 Dataset Overview, 🧩 Columns, 📈 Descriptive Statistics
- Include bullet points with bold labels (e.g., **Total Records (Rows):**)
- Create markdown tables with proper alignment
- Use checkmarks (✅) for positive indicators
- End with "Would you like me to:" followed by numbered options
- Write in a friendly, conversational ChatGPT-style tone
- Extract actual values from the code output/tool results - don't use placeholders
{"- Describe any visualizations that were generated" if has_image else ""}

Write the summary NOW - do not use tools, just write the text directly."""
                            
                            # Use the text-only model for generating summaries (no tools available)
                                # Create a simple prompt that includes the tool result and asks for summary
                                summary_prompt = f"""The Python code has been executed. Here are the results:

{tool_result_preview}

Based on the code output above, provide a CONCISE text summary with insights. This is the MAIN response to the user's request.

Start with "🤖 DataGem:" and keep it MEDIUM LENGTH - be thorough but to the point.

Include:

1. **Dataset Overview** (📊) - 2-3 bullet points:
   - Total records and columns
   - Data quality (missing values, etc.)

2. **Formatted Markdown Tables** (REQUIRED - extract from code output):
   - Columns table: | Column | Type | Description |
   - Descriptive statistics table: | Feature | Mean | Std Dev | Min | 25% | 50% | 75% | Max |
   - MUST extract actual values from the code output above - do not use placeholders

3. **Key Insights** (2-3 brief points):
   - Most interesting patterns or findings
   - Notable relationships

4. **Suggested Next Steps** (3-4 specific questions):
   - Actionable questions like "What are the correlations between columns?" or "Create a visualization of X"

Keep the summary concise and focused. Parse the code output to extract actual values for the tables."""
                            
                            # Use the text-only model (no tools) to generate the summary
                            followup_response = self.text_model.generate_content(
                                summary_prompt,
                                stream=True
                            )
                            
                            # Process followup response - this contains the text summary
                            followup_has_text = False
                            followup_text_accumulated = ""
                            print(f"🔄 Processing followup response for summary generation...")
                            
                            try:
                                event_count = 0
                                for followup_event in followup_response:
                                    event_count += 1
                                    
                                    # generate_content with stream=True returns chunks with text directly
                                    try:
                                        # Try direct text access first (most common for generate_content)
                                        if hasattr(followup_event, "text") and followup_event.text:
                                            has_output = True
                                            followup_has_text = True
                                            followup_text_accumulated += followup_event.text
                                            ai_response_content += followup_event.text
                                            yield followup_event.text
                                            continue
                                    except Exception as e:
                                        print(f"⚠️ Error accessing followup_event.text: {e}")
                                    
                                    # Try candidates path (for compatibility)
                                    try:
                                        if hasattr(followup_event, "candidates") and followup_event.candidates:
                                            for cand in followup_event.candidates:
                                                if hasattr(cand, "content") and cand.content:
                                                    if hasattr(cand.content, "parts"):
                                                        for p in cand.content.parts:
                                                            # Skip function calls, only process text
                                                            if hasattr(p, "function_call") and p.function_call:
                                                                print(f"⚠️ Warning: Gemini tried to call a function in followup response. Skipping function call.")
                                                                continue
                                                            if hasattr(p, "text") and p.text:
                                                                has_output = True
                                                                followup_has_text = True
                                                                followup_text_accumulated += p.text
                                                                ai_response_content += p.text
                                                                yield p.text
                                    except Exception as e:
                                        print(f"⚠️ Error accessing followup_event.candidates: {e}")
                                    
                                    # Try parts directly (for compatibility)
                                    try:
                                        if hasattr(followup_event, "parts"):
                                            for part in followup_event.parts:
                                                # Skip function calls, only process text
                                                if hasattr(part, "function_call") and part.function_call:
                                                    print(f"⚠️ Warning: Gemini tried to call a function in followup response. Skipping function call.")
                                                    continue
                                                if hasattr(part, "text") and part.text:
                                                    has_output = True
                                                    followup_has_text = True
                                                    followup_text_accumulated += part.text
                                                    ai_response_content += part.text
                                                    yield part.text
                                    except Exception as e:
                                        print(f"⚠️ Error accessing followup_event.parts: {e}")
                                
                                print(f"📊 Processed {event_count} followup events, has_text: {followup_has_text}, accumulated: {len(followup_text_accumulated)} chars")
                            except Exception as followup_error:
                                print(f"❌ Error processing followup response: {followup_error}")
                                traceback.print_exc()
                            
                            # If no followup text was generated, try to get the full response
                            if not followup_has_text:
                                print("⚠️ No followup text found in stream, trying to get full response...")
                                try:
                                    # Try to get the response object directly
                                    if hasattr(followup_response, "text"):
                                        text = followup_response.text
                                        if text:
                                            has_output = True
                                            followup_has_text = True
                                            ai_response_content += text
                                            yield text
                                    elif hasattr(followup_response, "candidates"):
                                        for cand in followup_response.candidates:
                                            if hasattr(cand, "content") and cand.content:
                                                if hasattr(cand.content, "parts"):
                                                    for p in cand.content.parts:
                                                        if hasattr(p, "text") and p.text:
                                                            has_output = True
                                                            followup_has_text = True
                                                            ai_response_content += p.text
                                                            yield p.text
                                except Exception as e:
                                    print(f"⚠️ Could not extract from followup response object: {e}")
                            
                            # If still no text, yield a helpful message
                            if not followup_has_text:
                                print("⚠️ Still no followup text generated, using default message")
                                default_summary = "\n\n## Summary\n\nThe analysis has been completed. Please review the code, output, and visualizations above for detailed results.\n\n**Note:** A detailed summary with tables and insights should have been generated. If you're seeing this message, please try asking your question again."
                                has_output = True
                                ai_response_content += default_summary
                                yield default_summary
                            else:
                                print(f"✅ Followup summary generated successfully ({len(followup_text_accumulated)} chars)")
                                print(f"📝 Summary preview (first 500 chars): {followup_text_accumulated[:500]}")
                        except Exception as tool_error:
                            error_msg = f"Error executing tool `{tool_name}`: {str(tool_error)}"
                            print(f"❌ Tool execution error: {tool_error}")
                            traceback.print_exc()
                            yield f"\n⚠️ {error_msg}\n"
                            ai_response_content += error_msg
                    
                    # --- If no function call found, try to extract text ---
                    elif not function_call_found:
                        try:
                            if hasattr(event, "text") and event.text:
                                has_output = True
                                ai_response_content += event.text
                                yield event.text
                        except Exception as text_error:
                            # Silently continue if text extraction fails
                            pass
                
                # Skip duplicate function call handling - already handled above
                # (Removed duplicate code block)
            
            except StopIteration:
                print("✅ Stream ended normally (StopIteration)")
            except Exception as stream_error:
                print(f"❌ Error during stream iteration: {stream_error}")
                traceback.print_exc()
                yield f"\n\n⚠️ Error processing stream: {str(stream_error)}\n"

            # --- If Gemini produced no text ---
            if not has_output:
                print("⚠️ No output from Gemini - trying to extract from candidates")
                # Try one more time to extract text from the response
                try:
                    # Get the last response to check for text
                    if hasattr(response_stream, "candidates"):
                        for candidate in response_stream.candidates:
                            if hasattr(candidate, "content") and candidate.content:
                                if hasattr(candidate.content, "parts"):
                                    for part in candidate.content.parts:
                                        if hasattr(part, "text") and part.text:
                                            has_output = True
                                            ai_response_content += part.text
                                            yield part.text
                except Exception as e:
                    print(f"⚠️ Could not extract from candidates: {e}")
                
                if not has_output:
                    # If we executed tools but got no text response, provide a helpful message
                    if ai_response_content:  # We have some content from tool execution
                        yield "\n\n✅ Analysis completed! The results are shown above.\n"
                        ai_response_content += "\n\n✅ Analysis completed!"
                    else:
                        yield "\n\n⚠️ I processed your request but encountered an issue. Let me try a different approach...\n"
                        # Try sending a simpler prompt
                        try:
                            simple_response = self.chat.send_message(
                                f"Please explain the results of the analysis in simple terms.",
                                stream=True
                            )
                            for simple_event in simple_response:
                                try:
                                    if hasattr(simple_event, "text") and simple_event.text:
                                        has_output = True
                                        ai_response_content += simple_event.text
                                        yield simple_event.text
                                except:
                                    if hasattr(simple_event, "candidates") and simple_event.candidates:
                                        for cand in simple_event.candidates:
                                            if hasattr(cand, "content") and cand.content:
                                                if hasattr(cand.content, "parts"):
                                                    for p in cand.content.parts:
                                                        if hasattr(p, "text") and p.text:
                                                            has_output = True
                                                            ai_response_content += p.text
                                                            yield p.text
                        except Exception as retry_error:
                            print(f"❌ Retry also failed: {retry_error}")
                            yield "\n\n❌ I'm having trouble generating a response. Please try rephrasing your question or check the backend logs for errors.\n"
                            ai_response_content = "Error generating response."

            print(f"✅ Stream completed. Total output length: {len(ai_response_content)} chars")

            # Save AI response
            if ai_response_content:
                crud.save_chat_message(
                    db=self.db,
                    user_id=self.user.id,
                    role="model",
                    content=ai_response_content
                )

        except Exception as e:
            print("❌ Exception during Gemini stream:")
            traceback.print_exc()
            error_message = f"❌ Gemini API Error: {str(e)}"
            yield error_message

            try:
                crud.save_chat_message(
                    db=self.db,
                    user_id=self.user.id,
                    role="model",
                    content=error_message
                )
            except Exception as save_error:
                print(f"❌ Failed to save error message: {save_error}")
