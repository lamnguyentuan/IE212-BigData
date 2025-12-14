#!/bin/bash

# ==========================================
# TikTok Harmful Content Detection - Auto Demo
# ==========================================

# 1. Reset Environment
echo "🔄 Resetting Environment..."
docker-compose down

if [ -f "downloaded_videos.txt" ]; then
    echo "🗑️ Removing downloaded_videos.txt..."
    rm downloaded_videos.txt
fi
rm -f /tmp/pipeline_ready
# Ensure port 8501 is free
pkill -f streamlit || true
fuser -k 8501/tcp || true

# 2. Start Infrastructure
echo "🚀 Starting Infrastructure (Docker)..."
docker-compose up -d

echo "⏳ Waiting 30s for services to initialize..."
sleep 30

# Stop Docker Dashboard to allow Local Dashboard (Terminal 4) to use port 8501
echo "🛑 Stopping Docker Dashboard to free port 8501..."
docker stop dashboard || true

# Initialize Kafka Topic to prevent Spark race condition
echo "🔧 Creating Kafka Topic 'video_events'..."
docker exec kafka kafka-topics --create --topic video_events --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 --if-not-exists || true

# Force Realtime Bucket
export MINIO_BUCKET=tiktok-realtime
echo "🎯 Set MINIO_BUCKET=$MINIO_BUCKET"

# 3. Check for Tmux
if ! command -v tmux &> /dev/null; then
    echo "⚠️ Tmux is not installed. Setting up manual instructions..."
    echo "Please open 3 terminals and run:"
    echo "1. python data_pipeline/auto_pipeline.py"
    echo "2. python data_pipeline/spark-streaming/main_stream.py --mode stream"
    echo "3. python data-ingestion/tiktok_crawl/tiktok_scraper.py --limit 5"
    exit 1
fi

# 4. Orchestrate with Tmux
SESSION="tiktok-demo"

# Kill existing session if any (to ensure fresh start)
tmux kill-session -t $SESSION 2>/dev/null
# Wait for server to cleanup
sleep 1

# Create new session (background)
tmux new-session -d -s $SESSION -n 'Demo'
if [ $? -ne 0 ]; then
    echo "❌ Failed to create tmux session. Please try running manual commands."
    exit 1
fi

# Function to setup a pane
setup_pane() {
    local pane_id=$1
    local title=$2
    local cmd=$3

    # Send activation command
    if [ -d "venv" ]; then
        tmux send-keys -t $pane_id "source venv/bin/activate" C-m
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        tmux send-keys -t $pane_id "source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate genai-env" C-m
    fi
    
    # Set title
    tmux select-pane -t $pane_id -T "$title"
    
    # Run command
    tmux send-keys -t $pane_id "$cmd" C-m
}

# --- Create Layout First ---
# Pane 0: Auto Pipeline
# Split -h creates Pane 1 (Right)
tmux split-window -h -t $SESSION:0
# Split -v from 0 creates Pane 2 (Bottom Left)
tmux split-window -v -t $SESSION:0
# Split -h from 2 creates Pane 3 (Bottom Right)
tmux split-window -h -t $SESSION:2

# Apply Layout
tmux select-layout -t $SESSION:0 tiled

# --- Launch Services ---

# Pane 0: Auto Pipeline
setup_pane "$SESSION:0.0" "Terminal 1: Auto Pipeline 🔄" "python data_pipeline/auto_pipeline.py"

# Pane 1: Spark Streaming
setup_pane "$SESSION:0.1" "Terminal 2: Spark Streaming ⚡" "python data_pipeline/spark-streaming/main_stream.py --mode stream"

# Pane 2: Crawler
setup_pane "$SESSION:0.2" "Terminal 3: Crawler 🕷️" "echo 'Waiting for Pipeline to be READY (/tmp/pipeline_ready)...'; while [ ! -f /tmp/pipeline_ready ]; do sleep 2; done; echo '✅ Pipeline Ready! Starting Crawler...'; export HEADLESS=false; python data-ingestion/tiktok_crawl/tiktok_scraper.py --limit 0 --explore"

# Pane 3: Dashboard
setup_pane "$SESSION:0.3" "Terminal 4: Dashboard 📊" "python -m streamlit run dashboard/app.py --server.port 8501"

# Select layout (Tiled)
tmux select-layout -t $SESSION:0 tiled

# Attach to session
echo "✅ Demo Started! Attaching to tmux..."
tmux attach -t $SESSION
