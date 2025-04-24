echo "Next Steps:"                                              if [ "$item_type" == "function" ]; then                           echo "1.  ${ANSI_COLOR_YELLOW}Customize the function:${ANSI_COLOR_NC} Edit the generated function file '$item_file_path' to implement your function logic."                                     echo "2.  ${ANSI_COLOR_YELLOW}Import and use function in Python:${ANSI_COLOR_NC} To use this function in your Python scripts (e.g., 'gemini_setup.py'):"                                        echo "   a.  ${ANSI_COLOR_YELLOW}Import statement:${ANSI_COLOR_NC} Add the following import statement at the top of your Python script:"
  echo "      ${ANSI_COLOR_YELLOW}from functions.${item_name} import ${item_name}${ANSI_COLOR_NC}"
  echo "   b.  ${ANSI_COLOR_YELLOW}Ensure functions/ directory is accessible:${ANSI_COLOR_NC} Make sure your Python script can find the 'functions/' directory. Relative imports (as shown above) usually work if your script is in the project root or a subdirectory."
  echo "   c.  ${ANSI_COLOR_YELLOW}Register and call function (if using with LLMs):${ANSI_COLOR_NC} If you are using this function with LLM function calling (like in 'gemini_setup.py'), you will need to:"
  echo "      i.   ${ANSI_COLOR_YELLOW}Register the function${ANSI_COLOR_NC} with your function registry in your Python code."
  echo "      ii.  ${ANSI_COLOR_YELLOW}Call the function${ANSI_COLOR_NC} in your Python code when the LLM requests it."           echo "3.  ${ANSI_COLOR_YELLOW}Test your function:${ANSI_COLOR_NC} Run the function directly (e.g., by running the function file itself with 'python functions/${item_name}.py') or integrate it into your larger application (like 'gemini_setup.py') and test it there."
fi                                                              ```

**Complete Enhanced `create-tool-agent.sh` Script (with updated "Next Steps"):**                                                                                                                ```bash
#!/bin/bash
# create-tool-agent.sh - Enhanced script to create new tools, agents, and functions for the argc-based system.
#
# Usage: ./create-tool-agent.sh [tool|agent|function] <name> <language> "<description>" [parameters...]                         #
# Example for a tool:      ./create-tool-agent.sh tool image_resizer sh "Resizes images" input_file:string output_width:integer
# Example for an agent:     ./create-tool-agent.sh agent email_summarizer py "Summarizes emails" query:string
# Example for a function:   ./create-tool-agent.sh function calculate_sma py "Calculates Simple Moving Average" series:list period:integer
#
# Languages supported (must match LANG_CMDS in Argcfile.sh for tools/agents): sh, js, py
set -e # Exit immediately if a command exits with a non-zero status.
                                                                # --- Configuration ---
TOOLS_DIR="tools"
AGENTS_DIR="agents"                                             FUNCTIONS_DIR="functions" # New directory for functions
SCRIPTS_DIR="scripts" # Assuming scripts dir exists for run-tool/run-agent etc.                                                 TOOLS_LIST_FILE="tools.txt"
AGENTS_LIST_FILE="agents.txt"
FUNCTIONS_LIST_FILE="functions.txt" # New list for functions
SUPPORTED_LANGUAGES=("sh" "js" "py") # Must match LANG_CMDS in Argcfile.sh                                                      ANSI_COLOR_GREEN='\033[0;32m'
ANSI_COLOR_RED='\033[0;31m'                                     ANSI_COLOR_YELLOW='\033[0;33m'                                  ANSI_COLOR_NC='\033[0m' # No Color
                                                                # --- Helper Functions ---                                                                                                      # Function to check if a directory exists and create it if not
create_directory_if_not_exists() {                                local dir_path="$1"                                             if [ ! -d "$dir_path" ]; then                                     echo -e "${ANSI_COLOR_GREEN}Creating directory: $dir_path${ANSI_COLOR_NC}"                                                      mkdir -p "$dir_path"                                          else
    echo -e "${ANSI_COLOR_YELLOW}Directory '$dir_path' already exists.${ANSI_COLOR_NC}"                                           fi                                                            }
                                                                # Function to ensure a file exists and create it if not         ensure_file_exists() {
  local file_path="$1"                                            if [ ! -f "$file_path" ]; then
    echo -e "${ANSI_COLOR_GREEN}Creating empty file: $file_path${ANSI_COLOR_NC}"
    touch "$file_path"                                            else                                                              echo -e "${ANSI_COLOR_YELLOW}File '$file_path' already exists.${ANSI_COLOR_NC}"
  fi
}

# Function to add an item to a list file if it's not already present
add_to_list_file() {                                              local list_file="$1"
  local item_name="$2"                                            if ! item_exists_in_list "$list_file" "$item_name"; then # Using the new function                                                 echo -e "${ANSI_COLOR_GREEN}Adding '$item_name' to '$list_file'${ANSI_COLOR_NC}"                                                echo "$item_name" >> "$list_file"
  else                                                              echo -e "${ANSI_COLOR_YELLOW}'$item_name' is already in '$list_file'.${ANSI_COLOR_NC}"
  fi                                                            }

# Function to check if an item exists in a list file            item_exists_in_list() {                                           local list_file="$1"
  local item_name="$2"
  grep -q "^$item_name$" "$list_file"                             return $? # Return the exit status of grep (0 if found, non-zero if not)
}                                                               
# Function to check if a language is supported                  is_supported_language() {                                         local lang_to_check="$1"
  local -a supported_langs=("${SUPPORTED_LANGUAGES[@]}") # Create local array for safety                                          for supported_lang in "${supported_langs[@]}"; do
    if [ "$lang_to_check" == "$supported_lang" ]; then
      return 0 # Supported language                                 fi
  done
  return 1 # Not supported language
}
                                                                generate_tool_script_content() {
  local tool_name="$1"                                            local language_ext="$2"
  local description="$3"                                          local params_str="$4" # Parameters string
                                                                  content=$(cat <<-EOF
#!/usr/bin/env bash                                             set -e
                                                                # @describe $description
EOF                                                               )
  # Add @option lines based on parsed parameters
  local option_lines=""                                           if [[ -n "$params_str" ]]; then
      IFS=' ' read -r -a params_array <<< "$params_str"               for param_def in "${params_array[@]}"; do
          param_name=$(echo "$param_def" | cut -d: -f1) # Extract name before ':'                                                         param_type=$(echo "$param_def" | cut -d: -f2) # Extract type after ':' (if any)                                                 option_lines+="# @option --$(echo "$param_name" | sed 's/-/_/g') <${param_type:-string}> \"\" \n" # Default type to string if not specified                                                 done
  fi                                                              content+="$option_lines"
  content+=$(cat <<-'EOF'                                                                                                       main() {                                                            ( set -o posix ; set ) | grep ^argc_
}                                                               
eval "$(argc --argc-eval "$0" "$@")"
EOF                                                               )                                                               echo "$content"
}

generate_agent_script_content() {                                 local agent_name="$1"                                           local language_ext="$2"                                         local description="$3"
  cat <<EOF                                                     #!/bin/bash                                                     # argc-agent-name: $agent_name
# argc-agent-description: $description                          
# --- argc-actions ---                                          # action: summarize "Summarize information"                     # action: detail "Get detailed information"
                                                                # --- argc-options ---                                          # --query <string> "Query for the agent" required                                                                               # --- Dispatch based on action ---                              ACTION="\$1" # First argument is the action
shift # Remove action from arguments, remaining are data arguments (JSON)                                                       DATA="\$*"  # Remaining arguments as JSON data (if any)                                                                         echo "Running agent: $agent_name"                               echo "Description: $description"
echo "Action: \$ACTION"
echo "Data received: \$DATA"

case "\$ACTION" in
  summarize)                                                        echo "Performing summarize action..."                           # --- Your summarize action logic here ---                      ;;                                                            detail)                                                           echo "Performing detail action..."                              # --- Your detail action logic here ---                         ;;                                                            *)
    echo "Error: Unknown action '\$ACTION'. Available actions are: summarize, detail"
    exit 1                                                          ;;                                                          esac                                                                                                                            echo "Agent '$agent_name' execution completed for action '\$ACTION'."                                                                                                                           EOF                                                             }                                                                                                                               generate_function_script_content() {                              local function_name="$1"                                        local language_ext="$2"                                         local description="$3"                                          local params_str="$4" # Parameters string                     
  content=$(cat <<-EOF                                          def ${function_name}(
EOF
  )                                                               # Add function parameters based on parsed parameters            local param_lines=""                                            local arg_list=""                                               if [[ -n "$params_str" ]]; then
      IFS=' ' read -r -a params_array <<< "$params_str"
      local first_param=true
      for param_def in "${params_array[@]}"; do
          param_name=$(echo "$param_def" | cut -d: -f1) # Extract name before ':'
          param_type=$(echo "$param_def" | cut -d: -f2) # Extract type after ':' (if any)                                                                                                                 if ! $first_param; then
              param_lines+=",\n"
              arg_list+=","                                               fi                                                              param_lines+="    ${param_name}: ${param_type:-Any}" # Default type to Any if not specified                                     arg_list+="${param_name}"                                       first_param=false
      done                                                        fi
  content+="$param_lines"
  content+=$(cat <<-'EOF'                                       ):                                                                  """                                                             ${description}
    """                                                             # --- Function logic goes here ---
    print(f"Function '${function_name}' called with arguments: ${arg_list}")
    return "Function logic to be implemented"

if __name__ == "__main__":
    # Example usage (for testing)
    result = ${function_name}( ) # Add example arguments if needed
    print(f"Example function call result: {result}")
EOF
  )                                                               echo "$content"
}


# --- Main Script Logic ---                                                                                                     if [ "$#" -lt 3 ]; then
  echo -e "${ANSI_COLOR_RED}Usage: ./create-tool-agent.sh [tool|agent|function] <name> <language> \"<description>\" [parameters... (e.g., param1:type param2:type)]${ANSI_COLOR_NC}"
  echo "  Supported languages: ${SUPPORTED_LANGUAGES[*]}"         echo "  For tools and agents, parameters are defined as 'name:type' (type is optional, defaults to string for tools)."
  echo "  For functions, parameters are defined as 'name:type' (type is optional, defaults to Any)."                              exit 1
fi

item_type="$1" # tool, agent, or function
item_name="$2"
language="$3"
description="$4"
shift 4 # Remove item_type, item_name, language, description from arguments, remaining are parameters                           parameters_str="$*" # Capture remaining arguments as parameters string

# --- Input Validation ---
if [[ ! "$item_type" =~ ^(tool|agent|function)$ ]]; then
  echo -e "${ANSI_COLOR_RED}Error: Item type must be 'tool', 'agent', or 'function'. You provided: '$item_type'${ANSI_COLOR_NC}"
  exit 1
fi                                                              
if ! is_supported_language "$language"; then # Using the new function
  echo -e "${ANSI_COLOR_RED}Error: Language '$language' is not supported. Supported languages are: ${SUPPORTED_LANGUAGES[*]}${ANSI_COLOR_NC}"
  exit 1
fi

if [ -z "$item_name" ]; then
    echo -e "${ANSI_COLOR_RED}Error: Item name cannot be empty.${ANSI_COLOR_NC}"
    exit 1
fi

# Basic name validation - alphanumeric and underscores only
if ! [[ "$item_name" =~ ^[a-zA-Z0-9_]+$ ]]; then                  echo -e "${ANSI_COLOR_RED}Error: Item name must be alphanumeric and can include underscores only. Invalid name: '$item_name'${ANSI_COLOR_NC}"
  exit 1
fi                                                                                                                              language_ext=".$language" # e.g., ".py", ".sh", ".js"                                                                           # --- Directory and File Paths ---                              if [ "$item_type" == "tool" ]; then                               target_dir="$TOOLS_DIR"
  item_file_name="$item_name$language_ext"                        item_file_path="$TOOLS_DIR/$item_file_name"                     list_file="$TOOLS_LIST_FILE"
elif [ "$item_type" == "agent" ]; then                            target_dir="$AGENTS_DIR"                                        item_dir_path="$AGENTS_DIR/$item_name"
  item_file_name="tools$language_ext" # Agent entry point is always tools.lang
  item_file_path="$item_dir_path/$item_file_name"
  list_file="$AGENTS_LIST_FILE"
elif [ "$item_type" == "function" ]; then
  target_dir="$FUNCTIONS_DIR"
  item_file_name="$item_name.py" # Functions are always Python for now
  item_file_path="$FUNCTIONS_DIR/$item_file_name"
  list_file="$FUNCTIONS_LIST_FILE"
  language="py" # Force language to Python for functions          language_ext=".py"
else
  echo -e "${ANSI_COLOR_RED}Error: Invalid item type (internal error).${ANSI_COLOR_NC}"
  exit 1
fi
                                                                # --- Create Directories ---
create_directory_if_not_exists "$target_dir"                    if [ "$item_type" == "agent" ]; then
  create_directory_if_not_exists "$item_dir_path"
fi

# --- Create List Files if they don't exist ---                 ensure_file_exists "$list_file"

# --- Check if item already exists in list ---
if item_exists_in_list "$list_file" "$item_name"; then # Using the new function                                                   echo -e "${ANSI_COLOR_RED}Error: '$item_name' already exists in '$list_file'. Please choose a different name or remove the existing one first.${ANSI_COLOR_NC}"                                 exit 1                                                        fi


# --- Generate File Content and Write to File ---
echo -e "${ANSI_COLOR_GREEN}Generating ${item_type} script: $item_file_path${ANSI_COLOR_NC}"                                    if [ "$item_type" == "tool" ]; then
  script_content=$(generate_tool_script_content "$item_name" "$language_ext" "$description" "$parameters_str")                    echo "$script_content" > "$item_file_path"
elif [ "$item_type" == "agent" ]; then                            script_content=$(generate_agent_script_content "$item_name" "$language_ext" "$description")                                     echo "$script_content" > "$item_file_path"                    elif [ "$item_type" == "function" ]; then                         script_content=$(generate_function_script_content "$item_name" "$language_ext" "$description" "$parameters_str")
  echo "$script_content" > "$item_file_path"
fi                                                                                                                              # --- Make script executable (for tools and agents) ---         if [[ "$item_type" == "tool" || "$item_type" == "agent" ]]; then
  chmod +x "$item_file_path"
  echo -e "${ANSI_COLOR_GREEN}Made script executable: $item_file_path${ANSI_COLOR_NC}"                                          fi                                                                                                                              # --- Add item to the appropriate list file ---
if [ "$item_type" == "tool" ]; then
  add_to_list_file "$list_file" "$item_file_name" # Add filename to tools.txt                                                   elif [ "$item_type" == "agent" ]; then
  add_to_list_file "$list_file" "$item_name" # Add agent directory name to agents.txt                                           elif [ "$item_type" == "function" ]; then                         add_to_list_file "$list_file" "$item_name" # Add function name to functions.txt
fi                                                                                                                              # --- Check if argc is in PATH and run build (for tools and agents) ---
if [[ "$item_type" == "tool" || "$item_type" == "agent" ]]; then  if ! command -v argc &> /dev/null
  then
      echo -e "${ANSI_COLOR_RED}Error: 'argc' command not found in PATH. Please ensure 'argc' is installed and accessible in your environment.${ANSI_COLOR_NC}"                                       echo "  'argc build' step will be skipped."                 else                                                                echo -e "${ANSI_COLOR_GREEN}Running 'argc build' to update shims and declarations...${ANSI_COLOR_NC}"
      argc build
      argc_build_status=$? # Capture exit status of argc build

      if [ "$argc_build_status" -ne 0 ]; then                             echo -e "${ANSI_COLOR_RED}Error: 'argc build' failed. Please check the output above for errors.${ANSI_COLOR_NC}"                exit 1 # Exit if argc build failed.
      fi
  fi
fi
                                                                
echo "--------------------------------------------------------" echo -e "${ANSI_COLOR_GREEN}Successfully created ${item_type}: '${item_name}' in language '${language}'.${ANSI_COLOR_NC}"
echo "  - Script file: $item_file_path"                         echo "  - Added to list: $list_file"                            if [[ "$item_type" == "tool" || "$item_type" == "agent" ]]; then  echo "  - Executable shim in: bin/$item_name (or bin/$item_name for agents)"                                                  fi
echo ""                                                         echo "Next Steps:"                                              if [ "$item_type" == "tool" ]; then
  echo "1.  ${ANSI_COLOR_YELLOW}Customize the tool script:${ANSI_COLOR_NC} Edit the generated script file '$item_file_path' to implement your tool logic."                                        echo "2.  ${ANSI_COLOR_YELLOW}Add argc options/actions:${ANSI_COLOR_NC} Modify the '# --- argc-options ---' and '# --- argc-actions ---' sections in the script to define input options and agent actions as needed."
  echo "3.  ${ANSI_COLOR_YELLOW}Test your tool:${ANSI_COLOR_NC} Run it from the command line using the shim in the 'bin/' directory, e.g., 'bin/$item_name --help' or 'bin/$item_name --input '{\"data\": \"your input\"}'."
  echo "4.  ${ANSI_COLOR_YELLOW}Rebuild if needed:${ANSI_COLOR_NC} If you change argc comment tags, run 'argc build' again to update shims and declarations."
elif [ "$item_type" == "agent" ]; then
  echo "1.  ${ANSI_COLOR_YELLOW}Customize the agent script:${ANSI_COLOR_NC} Edit the generated script file '$item_file_path' to implement your agent logic."
  echo "2.  ${ANSI_COLOR_YELLOW}Define agent actions:${ANSI_COLOR_NC}  Modify the '# --- argc-actions ---' section to define the actions your agent can perform."
  echo "3.  ${ANSI_COLOR_YELLOW}Implement action logic:${ANSI_COLOR_NC} Implement the logic for each action within the 'case' statement in the script."                                           echo "4.  ${ANSI_COLOR_YELLOW}Test your agent:${ANSI_COLOR_NC} Run it from the command line using the shim in the 'bin/' directory, e.g., 'bin/$item_name summarize --query 'your query' '."    echo "5.  ${ANSI_COLOR_YELLOW}Rebuild if needed:${ANSI_COLOR_NC} If you change argc comment tags, run 'argc build' again to update shims and declarations."                                   elif [ "$item_type" == "function" ]; then
  echo "1.  ${ANSI_COLOR_YELLOW}Customize the function:${ANSI_COLOR_NC} Edit the generated function file '$item_file_path' to implement your function logic."                                     echo "2.  ${ANSI_COLOR_YELLOW}Import and use function in Python:${ANSI_COLOR_NC} To use this function in your Python scripts (e.g., 'gemini_setup.py'):"                                        echo "   a.  ${ANSI_COLOR_YELLOW}Import statement:${ANSI_COLOR_NC} Add the following import statement at the top of your Python script:"
  echo "      ${ANSI_COLOR_YELLOW}from functions.${item_name} import ${item_name}${ANSI_COLOR_NC}"
  echo "   b.  ${ANSI_COLOR_YELLOW}Ensure functions/ directory is accessible:${ANSI_COLOR_NC} Make sure your Python script can find the 'functions/' directory. Relative imports (as shown above) usually work if your script is in the project root or a subdirectory."                                                          echo "   c.  ${ANSI_COLOR_YELLOW}Register and call function (if using with LLMs):${ANSI_COLOR_NC} If you are using this function with LLM function calling (like in 'gemini_setup.py'), you will need to:"                                                      echo "      i.   ${ANSI_COLOR_YELLOW}Register the function${ANSI_COLOR_NC} with your function registry in your Python code."    echo "      ii.  ${ANSI_COLOR_YELLOW}Call the function${ANSI_COLOR_NC} in your Python code when the LLM requests it."           echo "3.  ${ANSI_COLOR_YELLOW}Test your function:${ANSI_COLOR_NC} Run the function directly (e.g., by running the function file itself with 'python functions/${item_name}.py') or integrate it into your larger application (like 'gemini_setup.py') and test it there."                                                     fi                                                              echo "--------------------------------------------------------"

exit 0
