#!/bin/bash
# Terragon Autonomous SDLC Startup Script
# Starts the autonomous value discovery and execution system

set -e

echo "🚀 Starting Terragon Autonomous SDLC System..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check for required dependencies
echo "🔍 Checking dependencies..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

# Check required Python packages
python3 -c "import yaml" 2>/dev/null || {
    echo "📦 Installing required Python packages..."
    pip3 install pyyaml schedule requests > /dev/null 2>&1 || {
        echo "❌ Error: Failed to install required packages"
        exit 1
    }
}

# Check for configuration
if [ ! -f ".terragon/config.yaml" ]; then
    echo "❌ Error: Terragon configuration not found"
    echo "💡 Run the setup first to initialize configuration"
    exit 1
fi

# Create necessary directories
mkdir -p .terragon/logs
mkdir -p .terragon/backups

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --daemon    Run in daemon mode (background)"
    echo "  --once      Run single discovery cycle and exit"
    echo "  --status    Show current status"
    echo "  --stop      Stop running daemon"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Run interactively"
    echo "  $0 --daemon     # Run in background"
    echo "  $0 --once       # Single discovery cycle"
    echo "  $0 --status     # Check status"
}

# Function to run discovery once
run_once() {
    echo "🔍 Running single value discovery cycle..."
    cd .terragon
    python3 scheduler.py --once
    echo "✅ Discovery cycle completed"
}

# Function to show status
show_status() {
    echo "📊 Terragon Autonomous SDLC Status:"
    
    if [ -f ".terragon/scheduler.pid" ]; then
        PID=$(cat .terragon/scheduler.pid)
        if ps -p $PID > /dev/null 2>&1; then
            echo "🟢 Status: RUNNING (PID: $PID)"
        else
            echo "🔴 Status: STOPPED (stale PID file)"
            rm -f .terragon/scheduler.pid
        fi
    else
        echo "🔴 Status: STOPPED"
    fi
    
    if [ -f ".terragon/value-metrics.json" ]; then
        echo "📈 Latest metrics:"
        python3 -c "
import json
try:
    with open('.terragon/value-metrics.json') as f:
        data = json.load(f)
    print(f'  📅 Last updated: {data.get(\"timestamp\", \"Never\")}')
    print(f'  🎯 Items discovered: {data.get(\"discovered_items\", 0)}')
    print(f'  ⭐ High priority: {data.get(\"high_priority_items\", 0)}')
    print(f'  📊 Avg score: {data.get(\"average_composite_score\", 0):.1f}')
except Exception as e:
    print(f'  ❌ Metrics unavailable: {e}')
"
    fi
    
    if [ -f "AUTONOMOUS_BACKLOG.md" ]; then
        echo "📋 Backlog status: $(grep -c "| [0-9]" AUTONOMOUS_BACKLOG.md 2>/dev/null || echo 0) items"
    fi
}

# Function to stop daemon
stop_daemon() {
    if [ -f ".terragon/scheduler.pid" ]; then
        PID=$(cat .terragon/scheduler.pid)
        if ps -p $PID > /dev/null 2>&1; then
            echo "🛑 Stopping Terragon daemon (PID: $PID)..."
            kill $PID
            sleep 2
            if ps -p $PID > /dev/null 2>&1; then
                echo "⚠️  Force killing daemon..."
                kill -9 $PID
            fi
            rm -f .terragon/scheduler.pid
            echo "✅ Daemon stopped"
        else
            echo "❌ Daemon not running (removing stale PID file)"
            rm -f .terragon/scheduler.pid
        fi
    else
        echo "❌ Daemon not running"
    fi
}

# Function to run daemon
run_daemon() {
    if [ -f ".terragon/scheduler.pid" ]; then
        PID=$(cat .terragon/scheduler.pid)
        if ps -p $PID > /dev/null 2>&1; then
            echo "❌ Daemon already running (PID: $PID)"
            exit 1
        else
            rm -f .terragon/scheduler.pid
        fi
    fi
    
    echo "🚀 Starting Terragon daemon..."
    
    # Start scheduler in background
    cd .terragon
    nohup python3 scheduler.py > logs/scheduler.log 2>&1 &
    DAEMON_PID=$!
    echo $DAEMON_PID > scheduler.pid
    
    # Wait a moment to check if it started successfully
    sleep 2
    if ps -p $DAEMON_PID > /dev/null 2>&1; then
        echo "✅ Daemon started successfully (PID: $DAEMON_PID)"
        echo "📊 Monitor logs: tail -f .terragon/logs/scheduler.log"
        echo "🛑 Stop daemon: $0 --stop"
    else
        echo "❌ Failed to start daemon"
        rm -f scheduler.pid
        exit 1
    fi
}

# Function to run interactively
run_interactive() {
    echo "🎯 Starting interactive mode..."
    echo "💡 Press Ctrl+C to stop"
    
    cd .terragon
    python3 scheduler.py
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --once)
        run_once
        exit 0
        ;;
    --status)
        show_status
        exit 0
        ;;
    --stop)
        stop_daemon
        exit 0
        ;;
    --daemon|-d)
        run_daemon
        exit 0
        ;;
    "")
        run_interactive
        ;;
    *)
        echo "❌ Unknown option: $1"
        show_help
        exit 1
        ;;
esac