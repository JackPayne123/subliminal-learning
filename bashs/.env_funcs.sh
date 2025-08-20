longrun() { 
    timestamp=$(date +"%Y%m%d_%H%M%S")
    cmdname=$(echo "$1" | tr " " "_")
    [ -z "$cmdname" ] && cmdname="run"
    "$@" 2>&1 | tee "${PWD##*/}_${cmdname}_${timestamp}.log" && echo -e "\a"
}

gitpush() {
    VENV_NAME=".venv"
    if [ -f "$VENV_NAME/.gh_token" ]; then
        GH_TOKEN=$(cat "$VENV_NAME/.gh_token")
        git remote set-url origin "https://${GH_TOKEN}@github.com/${PWD##*/}.git"
    fi
    datetime=$(date +"%Y-%m-%d_%H-%M-%S")
    git add .
    git commit -m "${datetime} commit"
    git push
    echo "✅ Changes pushed with message: ${datetime} commit"
}
