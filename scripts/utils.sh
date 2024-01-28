# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# Get stacktrace (from https://gist.github.com/akostadinov/33bb2606afe1b334169dfbf202991d36)
function stack_trace() {
    local -a stack=("Bash Traceback:")
    local stack_size=${#FUNCNAME[@]}
    local -i i
    # we start with 2 to skip the stack trace function and the error handler function
    SKIP=2
    for (( i = $SKIP; i < stack_size; i++ )); do
	local func="${FUNCNAME[$i]:-(top level)}"
	local -i line="${BASH_LINENO[$(( i - 1 ))]}"
	local src="${BASH_SOURCE[$i]:-(no file)}"
	stack+=("    ($(( i - $SKIP ))) $func $src:$line")
    done
    (IFS=$'\n'; echo "${stack[*]}")
}

# Function to exit with generic error message
function exit_with_error {
    # call error function with predefined message
    error "Error in previous command. Exiting."
}

# Function to exit with error message and stack trace
function error {
    # print input with ansi codes for bold and red color
    echo -e "\033[1m\033[38;5;160m$*\033[0m"
    echo -e "\033[1m\033[38;5;160m`stack_trace`\033[0m"
    exit 1
}

# Function to print headers
function header {
    # print input with ansi codes for green color and horizontal rule
    terminal_width=$(tput cols)
    input_width=$((${#1} + 2))  # compute input width mod terminal width
    input_width=$((input_width % terminal_width))
    padding=$(( (terminal_width - input_width) / 2 ))
    padding=$(($padding < 0 ? $terminal_width : $padding))
    rule_char="─"
    rule=`printf "%${padding}s" | sed "s/ /$rule_char/g"`
    echo
    echo -e "\033[32m$rule $1 $rule\033[0m"
    echo
}

# Function to print info
function info {
    # print timestamp and input
    echo "`timestamp` $*" && echo
}

# Function to format paths for printing
function path {
    # print input with ansi codes for steel blue color
    echo -e "\033[94m$1\033[0m"
}

# Function to print commands
function cmd {
    # print input with ansi codes for steel blue color and padding on the left
    echo -e "\033[94m   $*\033[0m"
}

# Function to generate formatted timestamp for printing
function timestamp {
    # print timestamp with ansi codes for italics and bright black color
    echo -e "\033[3m\033[90m[`date +'%Y-%m-%d %H:%M:%S'`]\033[0m"
}
