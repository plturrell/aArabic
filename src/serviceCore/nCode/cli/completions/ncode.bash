#!/bin/bash
# Bash completion for nCode CLI

_ncode_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Commands
    local commands="index search query definition references symbols export health interactive help"
    
    # Options
    local opts="--server --qdrant --memgraph --verbose --json --help"
    
    # Complete commands
    if [ $COMP_CWORD -eq 1 ]; then
        COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
        return 0
    fi
    
    # Command-specific completion
    case "${COMP_WORDS[1]}" in
        index)
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
        search)
            return 0
            ;;
        query)
            return 0
            ;;
        definition|references)
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
        symbols)
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
        export)
            if [ $COMP_CWORD -eq 2 ]; then
                COMPREPLY=( $(compgen -W "json csv graphml" -- ${cur}) )
            else
                COMPREPLY=( $(compgen -f -- ${cur}) )
            fi
            return 0
            ;;
        *)
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
            ;;
    esac
}

complete -F _ncode_completion ncode
