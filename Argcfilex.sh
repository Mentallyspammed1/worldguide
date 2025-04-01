```bash                                                         #!/usr/bin/env bash                                             set -e
                                                                # @meta dotenv
                                                                BIN_DIR=bin
TMP_DIR="cache/__tmp__"
VENV_DIR=".venv"                                                                                                                LANG_CMDS=( \                                                       "sh:bash" \
    "js:node" \
    "py:python" \                                                   "java:java" # Added java placeholder - ensure java scripts exist if using java                                              )
                                                                # @cmd Run the tool                                             # @option -C --cwd <dir> Change the current working directory   # @alias tool:run
# @arg tool![`_choice_tool`] The tool name                      # @arg json The json data
run@tool() {                                                        if [[ -z "$argc_tool" ]]; then                                      _die "ERROR: Tool name is required. Usage: argc run@tool <tool_name> [json]"
    fi                                                              if [[ -z "$argc_json" ]]; then                                      declaration="$(generate-declarations@tool "$argc_tool" | jq -r '.[0]')"                                                         if [[ -n "$declaration" ]]; then
            _ask_json_data "$declaration"
        fi                                                          fi
    if [[ -z "$argc_json" ]]; then                                      lang="${argc_tool##*.}"                                         if [[ "$lang" == "py" ]]; then                                      _die "ERROR (Python Tool): No JSON data provided or generated. Please provide JSON data or ensure Python tool declarations are valid."                                                      elif [[ "$lang" == "java" ]]; then
            _die "ERROR (Java Tool): No JSON data provided or generated. Please provide JSON data or ensure Java tool declarations are valid."
        else
            _die "ERROR: No JSON data provided or generated.  Please provide JSON data or ensure declarations are valid."               fi                                                          fi
    lang="${argc_tool##*.}"                                         cmd="$(_lang_to_cmd "$lang")"                                   run_tool_script="scripts/run-tool.$lang"                        [[ -n "$argc_cwd" ]] && cd "$argc_cwd"
    echo "INFO: Running tool '$argc_tool' (lang: $lang) with script '$run_tool_script'..." # Added language to INFO
    exec "$cmd" "$run_tool_script" "$argc_tool" "$argc_json"
}                                                                                                                               # @cmd Generate declarations for a tool
# @alias tool:gen-decls
# @arg tool![`_choice_tool`] The tool name                      generate-declarations@tool() {                                      if [[ -z "$argc_tool" ]]; then                                      _die "ERROR: Tool name is required. Usage: argc generate-declarations@tool <tool_name>"                                     fi
    lang="${argc_tool##*.}"                                         cmd="$(_lang_to_cmd "$lang")"                                   build_declarations_script="scripts/build-declarations.$lang"    echo "INFO: Generating declarations for tool '$argc_tool' (lang: $lang) using script '$build_declarations_script'..." # Added language to INFO                                                  exec "$cmd" "$build_declarations_script" "$argc_tool"
}                                                                                                                                                                                               # @cmd Run the agent                                            # @alias agent:run
# @option -C --cwd <dir> Change the current working directory
# @arg agent![`_choice_agent`] The agent name                   # @arg action![?`_choice_agent_action`] The agent action        # @arg json The json data                                       run@agent() {                                                       if [[ -z "$argc_agent" ]]; then                                     _die "ERROR: Agent name is required. Usage: argc run@agent <agent_name> [action] [json]"                                    fi                                                              if [[ -z "$argc_action" ]]; then
        _die "ERROR: Agent action is required. Usage: argc run@agent <agent_name> <action> [json]" # More specific error            fi                                                              if [[ -z "$argc_json" ]]; then                                      declaration="$(generate-declarations@agent "$argc_agent" | jq --arg name "$argc_action" '.[] | select(.name == $name)')"        if [[ -n "$declaration" ]]; then                                    _ask_json_data "$declaration"                               fi                                                          fi
    if [[ -z "$argc_json" ]]; then                                      tools_path="$(_get_agent_tools_path "$argc_agent")"             lang="${tools_path##*.}"
        if [[ "$lang" == "py" ]]; then                                      _die "ERROR (Python Agent): No JSON data provided or generated. Please provide JSON data or ensure Python agent declarations for action '$argc_action' are valid."
        elif [[ "$lang" == "java" ]]; then                                  _die "ERROR (Java Agent): No JSON data provided or generated. Please provide JSON data or ensure Java agent declarations for action '$argc_action' are valid."                              else
            _die "ERROR: No JSON data provided or generated. Please provide JSON data or ensure declarations are valid for action '$argc_action'." # More specific error
        fi                                                          fi                                                              tools_path="$(_get_agent_tools_path "$argc_agent")"
    lang="${tools_path##*.}"                                        cmd="$(_lang_to_cmd "$lang")"                                   run_agent_script="scripts/run-agent.$lang"                      [[ -n "$argc_cwd" ]] && cd "$argc_cwd"                          echo "INFO: Running agent '$argc_agent', action '$argc_action' (lang: $lang) with script '$run_agent_script'..." # Added language to INFO
    exec "$cmd" "$run_agent_script"  "$argc_agent" "$argc_action" "$argc_json"
}                                                               

# @cmd Generate declarations for an agent                       # @alias agent:gen-decls
# @arg agent![`_choice_agent`] The agent name
generate-declarations@agent() {
    if [[ -z "$argc_agent" ]]; then                                     _die "ERROR: Agent name is required. Usage: argc generate-declarations@agent <agent_name>"
    fi
    tools_path="$(_get_agent_tools_path "$argc_agent")"
    lang="${tools_path##*.}"
    cmd="$(_lang_to_cmd "$lang")"                                   build_declarations_script="scripts/build-declarations.$lang"
    echo "INFO: Generating declarations for agent '$argc_agent' (lang: $lang) using script '$build_declarations_script'..." # Added language to INFO
    exec "$cmd" "$build_declarations_script" "$argc_agent"
}                                                               

_lang_to_cmd() {                                                    lang="$1"                                                       for entry in "${LANG_CMDS[@]}"; do
        IFS=: read -r l cmd <<< "$entry"                                if [[ "$l" == "$lang" ]]; then                                      echo "$cmd"
            return 0
        fi                                                          done
    _die "ERROR: Language '$lang' not supported in LANG_CMDS."  }
                                                                _get_agent_tools_path() {
    agent_name="$1"
    agent_base="${agent_name%.*}" # Remove extension if present
    echo "agents/${agent_base}/${agent_name}"                   }

                                                                _build_win_shim() {                                                 tool_name="$1"
    script_path="$2"
    shim_path="$BIN_DIR/$tool_name.cmd"
    mkdir -p "$BIN_DIR"                                             cat > "$shim_path" <<EOF
@echo off                                                       "%VENV_DIR%/Scripts/python.exe" "$script_path" %*
EOF                                                                 chmod +x "$shim_path"
    echo "INFO: Created Windows shim for '$tool_name' at '$shim_path'"
}
                                                                _build_py_shim() {
    tool_name="$1"                                                  script_path="$2"
    shim_path="$BIN_DIR/$tool_name"                                 mkdir -p "$BIN_DIR"
    cat > "$shim_path" <<EOF
#!/usr/bin/env bash                                             # shim for $(basename "$script_path")
VENV_DIR="\$(dirname "\$(dirname "\$(readlink -f "\$0")")")/.venv"
if [[ -f "\$VENV_DIR/bin/python" ]]; then                           "\$VENV_DIR/bin/python" "$script_path" "\$@"
else
    python "$script_path" "\$@"
fi
EOF                                                                 chmod +x "$shim_path"
    echo "INFO: Created python shim for '$tool_name' at '$shim_path'"                                                           }                                                                                                                               
_check_bin() {                                                      if [[ ! -d "$BIN_DIR" ]]; then
        echo "INFO: Creating bin directory '$BIN_DIR'"
        mkdir -p "$BIN_DIR"                                         fi
}
                                                                _check_envs() {
    if [[ ! -d "$VENV_DIR" ]]; then                                     echo "INFO: Creating virtual environment in '$VENV_DIR'"        python -m venv "$VENV_DIR"
    fi
}                                                               
_link_tool() {                                                      tool_path="$1"
    target_dir="$2" # BIN_DIR or AGENT_BIN_DIR
    tool_name=$(basename "$tool_path")                              link_path="$target_dir/$tool_name"
                                                                    if [[ -L "$link_path" ]]; then
        echo "INFO: Symlink '$link_path' already exists, skipping."
        return 0                                                    fi
                                                                    if [[ -e "$link_path" ]]; then
        echo "WARNING: File '$link_path' exists and is not a symlink. Please remove it if you want to create a symlink."
        return 1                                                    fi
                                                                    ln -s "$tool_path" "$link_path"
    chmod +x "$link_path" # Ensure it's executable after linking
    echo "INFO: Created symlink '$link_path' -> '$tool_path'"
    return 0                                                    }

                                                                _ask_json_data() {                                                  declaration_json="$1"                                           schema=$(jq -r '.schema' <<< "$declaration_json")               description=$(jq -r '.description' <<< "$declaration_json")     name=$(jq -r '.name' <<< "$declaration_json")               
    if [[ -n "$description" ]]; then                                    echo "Declaration for: '$name'"                                 echo "Description: $description"                            else                                                                echo "Declaration for: '$name'"
    fi
    if [[ -n "$schema" ]]; then                                         echo "Schema:"
        jq .schema <<< "$declaration_json"                          fi                                                                                                                              read -r -p "Enter JSON data (or leave empty to cancel): " user_json                                                             if [[ -n "$user_json" ]]; then
        argc_json="$user_json"                                      else
        echo "INFO: No JSON data provided, cancelling."
    fi                                                          }

                                                                _declarations_json_data() {
    declarations_dir="declarations"                                 mkdir -p "$declarations_dir"                                    declarations_file="$declarations_dir/$1.json"                   if [[ -f "$declarations_file" ]]; then                              cat "$declarations_file"
    else
        echo "[]" # Default to empty array if no declarations file
    fi
}                                                                                                                               
_normalize_path() {                                                 echo "$(realpath "$1")"
}
                                                                _is_win() {                                                         if [[ "$(uname -s)" == "MINGW"* ]]; then
        return 0 # True on Windows (Git Bash, etc.)                 else
        return 1 # False on Linux, macOS, etc.                      fi                                                          }                                                               
_argc_before() {                                                    local current_command="$1"
    shift
    local remaining_args=("$@")                                                                                                     local argc_index=-1                                             for i in "${!remaining_args[@]}"; do
        if [[ "${remaining_args[i]}" == "argc" ]]; then                     argc_index="$i"                                                 break                                                       fi                                                          done
                                                                    if [[ "$argc_index" -ne -1 ]]; then
        for ((i=argc_index+1; i<${#remaining_args[@]}; i++)); do            if [[ "${remaining_args[i]}" == "$current_command" ]]; then
                echo "true"                                                     return 0                                                    fi                                                          done
    fi                                                              echo "false"                                                    return 1                                                    }
                                                                                                                                _choice_tool() {                                                    local tools_dir="tools"
    find "$tools_dir" -maxdepth 2 -type f -executable -name "*.sh" -o -name "*.py" -o -name "*.js" -o -name "*.java" 2>/dev/null | while IFS= read -r tool_path; do                                     tool_name=$(basename "$tool_path")                              echo "$tool_name"                                           done                                                        }                                                               
_choice_web_search() {                                              echo "duckduckgo"
    echo "google"                                               }

_choice_code_interpreter() {                                        echo "python"
}                                                                                                                               _choice_agent() {                                                   local agents_dir="agents"                                       find "$agents_dir" -maxdepth 2 -type f -executable -name "*.sh" -o -name "*.py" -o -name "*.js" -o -name "*.java" 2>/dev/null | while IFS= read -r agent_path; do
        agent_name=$(basename "$agent_path")
        echo "$agent_name"
    done                                                        }

_choice_agent_action() {                                            if [[ -z "$argc_agent" ]]; then
        return 1 # No agent specified, no actions to suggest
    fi                                                              agent_name="$argc_agent"
    declarations_json="$(_declarations_json_data "$agent_name")"
    jq -r '.[] | .name' <<< "$declarations_json"                }

_choice_mcp_args() {                                                echo "-C ."
    echo "-C .. "
    echo "-C ../.."                                             }                                                                                                                               
_die() {
    echo "$@" >&2
    exit 1
}


eval "$(argc --argc-eval "$0" "$@")"
