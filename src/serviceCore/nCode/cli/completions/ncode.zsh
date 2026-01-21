#compdef ncode

# Zsh completion for nCode CLI

_ncode() {
    local -a commands
    commands=(
        'index:Index a project or file'
        'search:Semantic search across code'
        'query:Run Cypher query on graph database'
        'definition:Find symbol definition'
        'references:Find symbol references'
        'symbols:List symbols in file'
        'export:Export index data'
        'health:Check server health'
        'interactive:Start interactive mode'
        'help:Show help message'
    )
    
    local -a export_formats
    export_formats=(
        'json:JSON format'
        'csv:CSV format'
        'graphml:GraphML format'
    )
    
    _arguments -C \
        '1: :->command' \
        '*: :->args'
    
    case $state in
        command)
            _describe 'command' commands
            ;;
        args)
            case $words[2] in
                index)
                    _files
                    ;;
                search)
                    _message 'search query'
                    ;;
                query)
                    _message 'cypher query'
                    ;;
                definition|references)
                    _message 'location (file:line:char)'
                    ;;
                symbols)
                    _files
                    ;;
                export)
                    if (( CURRENT == 3 )); then
                        _describe 'format' export_formats
                    else
                        _files
                    fi
                    ;;
            esac
            ;;
    esac
}

_ncode "$@"
