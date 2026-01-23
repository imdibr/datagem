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

🚫 DO NOT USE THIS TOOL FOR:
- Greetings: "hi", "hey", "hello", "good morning", "good evening", "good afternoon"
- General questions: "what can you do?", "what all can you do?", "who are you?", "what is DataGem?"
- Casual conversation: "thanks", "how are you?", "tell me about yourself"
- Questions that don't explicitly request data analysis, statistics, visualizations, or code execution

✅ ONLY USE THIS TOOL WHEN:
- User explicitly asks for data analysis, visualization, statistics, or code execution
- User requests: "create a chart", "show me statistics", "analyze data", "plot", "visualize", etc.
- User wants to process, analyze, or visualize their dataset

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

⚠️ CRITICAL DECISION: Is this a data analysis question or a general conversation?

FOR GENERAL CONVERSATION (respond naturally, NO tools):
✅ Greetings: "hi", "hey", "hello", "good morning", "good evening", "good afternoon", "how are you", "what's up"
✅ Questions about you: "what can you do?", "what all can you do?", "who are you?", "what is DataGem?", "tell me about yourself"
   ✅ General chat: "thanks", "thank you", "goodbye", "bye", "okay", "cool"
✅ Casual conversation that doesn't mention data, analysis, statistics, charts, or visualizations
→ For these: Respond naturally and conversationally like a normal chatbot. Do NOT use any tools. Just chat!
→ Be friendly, helpful, and engaging. Mention your capabilities if asked, but don't run any code.

FOR DATA ANALYSIS QUESTIONS (use run_python_code tool):
✅ Questions about: data, dataset, statistics, analysis, visualize, plot, chart, graph, correlation, heatmap, box plot, scatter plot
✅ Requests to: create, show, display, analyze, calculate, find data insights
→ For these: YOU MUST use the run_python_code tool to execute Python code

CRITICAL RULES FOR DATA ANALYSIS:
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

RESPONSE FORMAT:
  - ALWAYS show the code you're running (it will be displayed automatically)
  - Run code using the tool
  - IMPORTANT: When showing DataFrames or statistics, use `print(df.to_markdown())` or `print(statistics.to_markdown())` to create formatted tables
  - After code execution, provide a CONCISE TEXT SUMMARY:
    * Direct answers - no fluff
    * Key findings only
    * Tables with actual data (extract from code output)
    * Brief insights, no redundancy

TEXT SUMMARY REQUIREMENTS:
- Write CONCISE summaries after code execution - be direct and to the point
- Structure:
  * Brief intro (1-2 sentences max)
  * Key findings/tables from code output
  * Brief insights (2-3 bullet points)
  * Optional: 1-2 next step suggestions
- Extract REAL values from code output - never use placeholders
- NO redundancy - don't repeat information already shown in code output
- Be direct and efficient - users want answers fast

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
                        "mode": "ANY",  # ANY enables constrained decoding for better function call handling
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

        # Check if prompt is conversational (improved detection)
        prompt_lower = prompt.lower().strip()
        # Single word greetings - always conversational
        single_word_greetings = ["hey", "hi", "hello", "hey!", "hi!", "hello!"]
        
        # Multi-word conversational phrases
        conversational_keywords = ["good morning", "good evening", "good afternoon", "good night",
                                 "what can you do", "what all can you do", "who are you", "tell me about yourself",
                                 "thanks", "thank you", "how are you", "what's up", "what's going on"]
        
        # Data analysis keywords - if present, it's NOT conversational
        data_keywords = ["data", "dataset", "analyze", "analysis", "statistics", "visualize", "visualization",
                        "plot", "chart", "graph", "correlation", "heatmap", "scatter", "box plot", "histogram",
                        "create a", "show me", "display", "calculate", "find", "insights", "summary report",
                        "pdf", "html", "model", "predict"]
        
        # Check if it's a single word greeting
        is_single_greeting = prompt_lower in single_word_greetings
        
        # Check if it contains conversational keywords
        has_conversational = any(keyword in prompt_lower for keyword in conversational_keywords)
        
        # Check if it contains data keywords
        has_data_keywords = any(keyword in prompt_lower for keyword in data_keywords)
        
        # Mark as conversational if:
        # 1. It's a single word greeting, OR
        # 2. It has conversational keywords AND no data keywords
        is_conversational = is_single_greeting or (has_conversational and not has_data_keywords)
        
        # Enhance prompt based on type
        enhanced_prompt = prompt
        if is_conversational:
            # For conversational prompts, create a friendly, natural response prompt
            dataset_info = ""
            if self.dataset and len(self.dataset) > 0:
                row_count = len(self.dataset)
                columns = list(self.dataset[0].keys()) if self.dataset else []
                dataset_info = f" I can see you have a dataset loaded with {row_count} rows and {len(columns)} columns. I'm ready to help you analyze it whenever you're ready!"
            
            enhanced_prompt = f"""User said: "{prompt}"

You are DataGem, a friendly AI assistant and expert data analyst. The user is just having a casual conversation or greeting you.

Respond naturally and conversationally:
- If it's "hey", "hi", or "hello": Greet them warmly and offer to help with data analysis{dataset_info}
- If they ask what you can do: Briefly explain that you're DataGem, an AI data analyst that can analyze datasets, create visualizations, find insights, build models, etc. Keep it concise and friendly.
- If they ask "who are you": Tell them you're DataGem, an AI assistant specialized in data analysis
- For any other casual conversation: Respond naturally, be friendly and helpful

Keep your response:
- Short and friendly (2-3 sentences max)
- Conversational and natural
- Helpful but not overwhelming
- End with an offer to help with data analysis if relevant

Remember: This is just a chat - no data analysis, no code, just a friendly conversation."""
            print(f"💬 Detected conversational prompt ('{prompt}') - responding naturally without tools")
        elif self.dataset and len(self.dataset) > 0:
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
            
            dataset_context = f"""User question: {prompt}

Dataset available:
- {row_count} rows, {len(columns)} columns
- All columns: {', '.join(columns)}
- Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None detected'}
- DataFrame name: 'df'

IMPORTANT: 
- Use run_python_code tool immediately to analyze or visualize
- For visualizations, save plots as base64 images using PLOT_IMG_BASE64 format
- After running code, provide a CONCISE text summary:
  * Key findings only - no redundancy
  * Brief tables with actual data
  * 2-3 insights maximum
  * Direct and actionable
- When asked for a summary, generate tables with key statistics and findings
- Format summaries like ChatGPT or Gemini with structured tables and clear explanations"""
            enhanced_prompt = dataset_context
            print(f"📊 Enhanced prompt with dataset context ({row_count} rows, {len(columns)} columns)")

        prompt_parts = [enhanced_prompt]
        if image:
            prompt_parts.append(image)

        ai_response_content = ""
        has_output = False
        iteration_count = 0

        try:
            # For conversational prompts, use text_model (no tools) to ensure no tool calls
            if is_conversational:
                print("💬 Using text-only model for conversational response (no tools available)")
                try:
                    # Use text_model which has no tools - this guarantees no tool calls
                    response_stream = self.text_model.generate_content(
                        enhanced_prompt,
                        stream=True
                    )
                    print("✅ Text-only stream created for conversational response")
                except Exception as stream_init_error:
                    print(f"❌ Error creating conversational stream: {stream_init_error}")
                    yield f"❌ Error initializing response: {str(stream_init_error)}"
                    return
            else:
                # For data analysis, use model with tools
                print("🔄 Starting Gemini stream with tools...")
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
                    
                    # --- Handle standard text chunks ---
                    try:
                        if hasattr(event, "text") and event.text:
                            has_output = True
                            ai_response_content += event.text
                            yield event.text
                            continue  # Skip to next event
                    except Exception as text_error:
                        # If accessing .text fails, it might be a function call or invalid response
                        # For conversational prompts, skip tool call handling
                        if is_conversational:
                            print(f"⚠️ Error accessing text in conversational response: {text_error}")
                            continue
                        
                        # Check if it's a function call issue
                        error_str = str(text_error)
                        if "finish_reason" in error_str or "FunctionCall" in error_str:
                            print(f"⚠️ Function call issue detected: {text_error}")
                            # Try to get candidate and check for function calls
                            try:
                                if hasattr(event, "candidates") and event.candidates:
                                    for candidate in event.candidates:
                                        if hasattr(candidate, "content") and candidate.content:
                                            if hasattr(candidate.content, "parts"):
                                                for part in candidate.content.parts:
                                                    if hasattr(part, "text") and part.text:
                                                        has_output = True
                                                        ai_response_content += part.text
                                                        yield part.text
                            except Exception as e:
                                print(f"⚠️ Could not extract text from candidate: {e}")
                        else:
                            print(f"⚠️ Could not access text: {text_error}")
                        pass

                    # --- Handle Gemini tool calls from event.parts ---
                    # Skip tool call handling for conversational prompts (text_model doesn't have tools)
                    if not is_conversational and hasattr(event, "parts") and event.parts:
                        for part in event.parts:
                            if hasattr(part, "function_call") and part.function_call:
                                fc = part.function_call
                                tool_name = fc.name if hasattr(fc, "name") else "unknown"
                                tool_args = dict(fc.args) if hasattr(fc, "args") and fc.args else {}
                                
                                # SAFEGUARD: Check if this was a conversational prompt
                                # Re-check the prompt to see if it's conversational
                                prompt_check = prompt.lower().strip()
                                single_greetings = ["hey", "hi", "hello", "hey!", "hi!", "hello!"]
                                is_conv_check = prompt_check in single_greetings or \
                                               (any(kw in prompt_check for kw in ["good morning", "good evening", "what can you do", "who are you"]) and 
                                                not any(kw in prompt_check for kw in ["data", "analyze", "plot", "chart", "create", "show"]))
                                
                                if is_conv_check:
                                    print(f"⚠️ BLOCKED: Attempted tool call '{tool_name}' for conversational prompt '{prompt}'. Ignoring.")
                                    # Skip this tool call and continue to get text response
                                    continue
                                
                                # Process the function call
                                print(f"🔧 Executing tool from event.parts: {tool_name}")
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
                                        followup_prompt = f"""Code execution error:

{tool_result_preview}

Provide a brief summary based on available dataset info. Be concise.

Format:
🤖 DataGem: [Brief explanation of error/what was attempted]

[Key dataset info if available]

**Next steps:** [1-2 suggestions]

Keep it short - text only, no tools."""
                                    else:
                                        followup_prompt = f"""Code executed successfully. Results:

{tool_result_preview}

Write a CONCISE summary. Be direct - no fluff, no redundancy.

Requirements:
- Extract key findings from code output above
- Show actual data/tables (use markdown tables)
- 2-3 brief insights only
- NO repetition of what's already in code output
- Keep it short and actionable

Format:
🤖 DataGem: [Brief 1-sentence intro]

[Key findings/tables from code output]

**Insights:**
- [1-2 bullet points]
- [1-2 bullet points]

{"**Visualization:** [Brief description if plot was generated]" if has_image else ""}

Start NOW - text only, no tools."""
                                    
                                    # Use the text-only model for generating summaries (no tools available)
                                    # Use the detailed followup_prompt for comprehensive summaries
                                    print(f"📝 Generating text summary from tool results...")
                                    summary_prompt = followup_prompt  # Use the comprehensive prompt
                                    
                                    # Use the text-only model (no tools) to generate the summary
                                    try:
                                        followup_response = self.text_model.generate_content(
                                            summary_prompt,
                                            stream=True
                                        )
                                        print(f"✅ Summary generation started")
                                    except Exception as summary_error:
                                        print(f"❌ Error starting summary: {summary_error}")
                                        # Fallback to simpler prompt
                                        summary_prompt = f"""Code executed. Results:

{tool_result_preview}

Write a brief summary: key findings only, no redundancy. Extract actual values from output. Be direct and concise."""
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
                                        default_summary = "\n\n✅ Analysis completed. Review the code, output, and visualizations above."
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
                                    
                                    # Set default values if exception occurred
                                    tool_result = error_msg
                                    code_failed = True
                                    tool_result_preview = error_msg
                                    has_image = False
                                    
                                    # Define followup_prompt even on error
                                    followup_prompt = f"""The tool execution encountered an error: {error_msg}

Provide a brief summary based on available dataset info. Be concise - text only, no tools."""
                                    
                                    # Still try to send followup response even on error
                                    try:
                                        # Use the text-only model for generating summaries (no tools available)
                                        summary_prompt = f"""The tool execution encountered an error, but please still provide insights based on the dataset information:

{followup_prompt}

Provide a brief summary with key insights. Start with "🤖 DataGem:" - be concise, no redundancy."""
                                        
                                        # Use the text-only model (no tools) to generate the summary
                                        followup_response = self.text_model.generate_content(
                                            summary_prompt,
                                            stream=True
                                        )
                                        
                                        # Process followup response
                                        followup_has_text = False
                                        followup_text_accumulated = ""
                                        
                                        try:
                                            for followup_event in followup_response:
                                                try:
                                                    if hasattr(followup_event, "text") and followup_event.text:
                                                        has_output = True
                                                        followup_has_text = True
                                                        followup_text_accumulated += followup_event.text
                                                        ai_response_content += followup_event.text
                                                        yield followup_event.text
                                                        continue
                                                except Exception as e:
                                                    print(f"⚠️ Error accessing followup_event.text: {e}")
                                                
                                                # Try candidates path
                                                try:
                                                    if hasattr(followup_event, "candidates") and followup_event.candidates:
                                                        for cand in followup_event.candidates:
                                                            if hasattr(cand, "content") and cand.content:
                                                                if hasattr(cand.content, "parts"):
                                                                    for p in cand.content.parts:
                                                                        if hasattr(p, "text") and p.text:
                                                                            has_output = True
                                                                            followup_has_text = True
                                                                            followup_text_accumulated += p.text
                                                                            ai_response_content += p.text
                                                                            yield p.text
                                                except Exception as e:
                                                    print(f"⚠️ Error accessing followup_event.candidates: {e}")
                                                
                                                # Try parts directly
                                                try:
                                                    if hasattr(followup_event, "parts"):
                                                        for part in followup_event.parts:
                                                            if hasattr(part, "text") and part.text:
                                                                has_output = True
                                                                followup_has_text = True
                                                                followup_text_accumulated += part.text
                                                                ai_response_content += part.text
                                                                yield part.text
                                                except Exception as e:
                                                    print(f"⚠️ Error accessing followup_event.parts: {e}")
                                        except Exception as followup_error:
                                            print(f"❌ Error processing followup response: {followup_error}")
                                            traceback.print_exc()
                                        
                                        if not followup_has_text:
                                            print("⚠️ No followup text generated after error, using default message")
                                            default_summary = "\n\n⚠️ Analysis encountered an error. Check the code output above for details."
                                            has_output = True
                                            ai_response_content += default_summary
                                            yield default_summary
                                    except Exception as followup_send_error:
                                        print(f"❌ Error sending followup after tool error: {followup_send_error}")
                                        traceback.print_exc()
                
                # --- Handle Gemini tool calls from candidates (if not already handled) ---
                # Skip tool call handling for conversational prompts (text_model doesn't have tools)
                # Only process if we haven't already handled function calls from event.parts
                if not is_conversational and hasattr(event, "candidates") and event.candidates and not hasattr(event, "parts"):
                    for candidate in event.candidates:
                        if hasattr(candidate, "content") and candidate.content:
                            if hasattr(candidate.content, "parts"):
                                for part in candidate.content.parts:
                                    if hasattr(part, "function_call") and part.function_call:
                                        fc = part.function_call
                                        tool_name = fc.name if hasattr(fc, "name") else "unknown"
                                        tool_args = dict(fc.args) if hasattr(fc, "args") and fc.args else {}
                                        
                                        # SAFEGUARD: Check if this was a conversational prompt
                                        prompt_check = prompt.lower().strip()
                                        single_greetings = ["hey", "hi", "hello", "hey!", "hi!", "hello!"]
                                        is_conv_check = prompt_check in single_greetings or \
                                                       (any(kw in prompt_check for kw in ["good morning", "good evening", "what can you do", "who are you"]) and 
                                                        not any(kw in prompt_check for kw in ["data", "analyze", "plot", "chart", "create", "show"]))
                                        
                                        if is_conv_check:
                                            print(f"⚠️ BLOCKED: Attempted tool call '{tool_name}' for conversational prompt '{prompt}'. Ignoring.")
                                            continue
                                        
                                        print(f"🔧 Executing tool from candidates: {tool_name}")
                                        yield f"\n\n🤖 **Executing:** `{tool_name}`\n\n"
                                        
                                        # Process the function call (same logic as above)
                                        try:
                                            # Initialize tool_result to ensure it's always defined
                                            tool_result = None
                                            
                                            if tool_name == "run_python_code":
                                                code = tool_args.get("code", "")
                                                if not code:
                                                    tool_result = "Error: No code provided to execute."
                                                else:
                                                    yield f"```python\n{code}\n```\n\n"
                                                    print(f"💻 Running Python code ({len(code)} chars)...")
                                                    tool_result = tools.run_python_code(code, self.dataset)
                                                    print(f"✅ Code execution completed")
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
                                            
                                            # Feed back to Gemini
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
                            
Write the summary NOW - do not use tools, just write the text directly."""
                                            else:
                                                followup_prompt = f"""Code executed successfully. Results:

{tool_result_preview}

Write a CONCISE summary. Extract key findings from output above. Be direct - no redundancy.

Format:
🤖 DataGem: [1-sentence intro]

[Key tables/data from output]

**Insights:** [2-3 bullet points max]

Keep it short and actionable. Text only."""
                                            
                                            # Use the text-only model (no tools) to generate the summary
                                            print(f"📝 Generating text summary from tool results...")
                                            summary_prompt = followup_prompt
                                            
                                            try:
                                                followup_response = self.text_model.generate_content(
                                                    summary_prompt,
                                                    stream=True
                                                )
                                                
                                                followup_has_text = False
                                                followup_text_accumulated = ""
                                                
                                                try:
                                                    for followup_event in followup_response:
                                                        try:
                                                            # Try direct text access
                                                            if hasattr(followup_event, "text") and followup_event.text:
                                                                has_output = True
                                                                followup_has_text = True
                                                                followup_text_accumulated += followup_event.text
                                                                ai_response_content += followup_event.text
                                                                yield followup_event.text
                                                                continue
                                                        except Exception as e:
                                                            print(f"⚠️ Error accessing followup_event.text: {e}")
                                                        
                                                        # Try candidates path
                                                        try:
                                                            if hasattr(followup_event, "candidates") and followup_event.candidates:
                                                                for cand in followup_event.candidates:
                                                                    if hasattr(cand, "content") and cand.content:
                                                                        if hasattr(cand.content, "parts"):
                                                                            for p in cand.content.parts:
                                                                                if hasattr(p, "text") and p.text:
                                                                                    has_output = True
                                                                                    followup_has_text = True
                                                                                    followup_text_accumulated += p.text
                                                                                    ai_response_content += p.text
                                                                                    yield p.text
                                                        except Exception as e:
                                                            print(f"⚠️ Error accessing followup_event.candidates: {e}")
                                                        
                                                        # Try parts directly
                                                        try:
                                                            if hasattr(followup_event, "parts"):
                                                                for part in followup_event.parts:
                                                                    if hasattr(part, "text") and part.text:
                                                                        has_output = True
                                                                        followup_has_text = True
                                                                        followup_text_accumulated += part.text
                                                                        ai_response_content += part.text
                                                                        yield part.text
                                                        except Exception as e:
                                                            print(f"⚠️ Error accessing followup_event.parts: {e}")
                                                except Exception as followup_error:
                                                    print(f"❌ Error processing followup response: {followup_error}")
                                                    traceback.print_exc()
                                            except Exception as generate_error:
                                                print(f"❌ Error generating summary: {generate_error}")
                                                yield f"\n❌ Error generating summary: {str(generate_error)}\n"
                                            
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
                                                default_summary = "\n\n✅ Analysis completed. Review the code, output, and visualizations above."
                                                has_output = True
                                                ai_response_content += default_summary
                                                yield default_summary
                                            else:
                                                print(f"✅ Followup summary generated successfully ({len(followup_text_accumulated)} chars)")
                                        except Exception as tool_error:
                                            error_msg = f"Error executing tool `{tool_name}`: {str(tool_error)}"
                                            print(f"❌ Tool execution error: {tool_error}")
                                            traceback.print_exc()
                                            yield f"\n⚠️ {error_msg}\n"
                                            ai_response_content += error_msg
                                            
                                            # Set default values if exception occurred
                                            tool_result = error_msg
                                            code_failed = True
                                            tool_result_preview = error_msg
                                            has_image = False
                                            
                                            # Define followup_prompt even on error
                                            followup_prompt = f"""The tool execution encountered an error: {error_msg}

Provide a brief summary based on available dataset info. Be concise - text only, no tools."""
                                            
                                            # Still try to send followup response even on error
                                            try:
                                                # Use the text-only model for generating summaries (no tools available)
                                                summary_prompt = f"""The tool execution encountered an error, but please still provide insights based on the dataset information:

{followup_prompt}

Provide a brief summary with key insights. Start with "🤖 DataGem:" - be concise, no redundancy."""
                                                
                                                # Use the text-only model (no tools) to generate the summary
                                                followup_response = self.text_model.generate_content(
                                                    summary_prompt,
                                                    stream=True
                                                )
                                                
                                                # Process followup response
                                                followup_has_text = False
                                                followup_text_accumulated = ""
                                                
                                                try:
                                                    for followup_event in followup_response:
                                                        try:
                                                            # Try direct text access
                                                            if hasattr(followup_event, "text") and followup_event.text:
                                                                has_output = True
                                                                followup_has_text = True
                                                                followup_text_accumulated += followup_event.text
                                                                ai_response_content += followup_event.text
                                                                yield followup_event.text
                                                                continue
                                                        except Exception as e:
                                                            print(f"⚠️ Error accessing followup_event.text: {e}")
                                                        
                                                        # Try candidates path
                                                        try:
                                                            if hasattr(followup_event, "candidates") and followup_event.candidates:
                                                                for cand in followup_event.candidates:
                                                                    if hasattr(cand, "content") and cand.content:
                                                                        if hasattr(cand.content, "parts"):
                                                                            for p in cand.content.parts:
                                                                                if hasattr(p, "text") and p.text:
                                                                                    has_output = True
                                                                                    followup_has_text = True
                                                                                    followup_text_accumulated += p.text
                                                                                    ai_response_content += p.text
                                                                                    yield p.text
                                                        except Exception as e:
                                                            print(f"⚠️ Error accessing followup_event.candidates: {e}")
                                                        
                                                        # Try parts directly
                                                        try:
                                                            if hasattr(followup_event, "parts"):
                                                                for part in followup_event.parts:
                                                                    if hasattr(part, "text") and part.text:
                                                                        has_output = True
                                                                        followup_has_text = True
                                                                        followup_text_accumulated += part.text
                                                                        ai_response_content += part.text
                                                                        yield part.text
                                                        except Exception as e:
                                                            print(f"⚠️ Error accessing followup_event.parts: {e}")
                                                except Exception as followup_error:
                                                    print(f"❌ Error processing followup response: {followup_error}")
                                                    traceback.print_exc()
                                                
                                                if not followup_has_text:
                                                    print("⚠️ No followup text generated after error, using default message")
                                                    default_summary = "\n\n⚠️ Analysis encountered an error. Check the code output above for details."
                                                    has_output = True
                                                    ai_response_content += default_summary
                                                    yield default_summary
                                            except Exception as followup_send_error:
                                                print(f"❌ Error sending followup after tool error: {followup_send_error}")
                                                traceback.print_exc()
            
            except StopIteration:
                print("✅ Stream ended normally (StopIteration)")
            except Exception as stream_error:
                print(f"❌ Error during stream iteration: {stream_error}")
                traceback.print_exc()
                yield f"\n\n⚠️ Error processing stream: {str(stream_error)}\n"

            # --- If Gemini produced no text ---
            if not has_output:
                print("⚠️ No output from Gemini - trying to extract from candidates")
                # For conversational prompts, this shouldn't happen often, but handle it gracefully
                if is_conversational:
                    # For conversational prompts, provide a friendly fallback
                    fallback_response = "Hi! I'm DataGem, your AI data analyst assistant. How can I help you today?"
                    if self.dataset and len(self.dataset) > 0:
                        row_count = len(self.dataset)
                        columns = list(self.dataset[0].keys()) if self.dataset else []
                        fallback_response = f"Hi! I'm DataGem, your AI data analyst. I can see you have a dataset with {row_count} rows and {len(columns)} columns loaded. How can I help you analyze it?"
                    has_output = True
                    ai_response_content = fallback_response
                    yield fallback_response
                else:
                    # For data analysis, try to extract from candidates
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
