# Totem

A proxy chatbot to answer professional and personal questions about yourself using AI. This application creates an interactive chat interface powered by OpenAI (or local LLM) that can answer questions about your background, career, and experience based on your LinkedIn profile and personal summary.

## Features

- 🤖 AI-powered chat interface using GPT-4o-mini or local Ollama models
- 📄 Automatically ingests your LinkedIn profile (PDF) and personal summary
- 📧 Collects interested visitor contact information
- 📱 Push notifications via Pushover for new contacts and unknown questions
- 🚀 Easy deployment to HuggingFace Spaces
- 💻 Local development support with Ollama

## Prerequisites

Before you begin, ensure you have:
- Oracle VirtualBox with Ubuntu installed
- Internet connection
- (Optional) Windows host with Ollama for local LLM testing
- (Optional) OpenAI API key for deployment
- (Optional) Pushover account for notifications

## Installation Guide

### Step 1: Install Cursor IDE on Ubuntu (VirtualBox)

1. **Download Cursor AppImage:**
   ```bash
   cd ~/Downloads
   wget https://downloader.cursor.sh/linux/appImage/x64
   mv x64 cursor.AppImage
   chmod +x cursor.AppImage
   ```

2. **Move to Applications directory (optional):**
   ```bash
   sudo mkdir -p /opt/cursor
   sudo mv cursor.AppImage /opt/cursor/
   ```

3. **Create a desktop entry (optional):**
   ```bash
   cat > ~/.local/share/applications/cursor.desktop << 'EOF'
   [Desktop Entry]
   Name=Cursor
   Exec=/opt/cursor/cursor.AppImage
   Terminal=false
   Type=Application
   Icon=cursor
   StartupWMClass=Cursor
   Comment=AI-powered code editor
   Categories=Development;IDE;
   EOF
   ```

4. **Launch Cursor:**
   ```bash
   /opt/cursor/cursor.AppImage
   # Or click the Cursor icon in your applications menu
   ```

### Step 2: Install Python Extension in Cursor

1. Open Cursor IDE
2. Click on the Extensions icon in the sidebar (or press `Ctrl+Shift+X`)
3. Search for "Python" by Microsoft (`ms-python.python`)
4. Click **Install**
5. Wait for the installation to complete
6. Restart Cursor if prompted

### Step 3: Install UV Package Manager

UV is a fast Python package and project manager. Install it using:

```bash
# On Ubuntu (using the standalone installer)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal or run:
```bash
source $HOME/.cargo/env
```

Verify the installation:
```bash
uv --version
```

### Step 4: Clone This Repository

```bash
# Navigate to your projects directory
cd ~/Projects
# Or create one if it doesn't exist
mkdir -p ~/Projects && cd ~/Projects

# Clone the repository
git clone https://github.com/maihuuthien/totem.git
cd totem
```

### Step 5: Setup Python Virtual Environment

UV automatically creates and manages virtual environments. Sync dependencies:

```bash
uv sync
```

This command will:
- Create a virtual environment (`.venv`)
- Install Python 3.12+ if needed
- Install all dependencies from `pyproject.toml`

### Step 6: Create Environment File

Create a `.env` file in the project root:

```bash
touch .env
```

Open `.env` in your editor and add your configuration:

```env
# Required for deployment (HuggingFace with OpenAI)
OPENAI_API_KEY=sk-proj-your-openai-api-key-here

# Optional: Pushover notifications
PUSHOVER_USER=your-pushover-user-key
PUSHOVER_TOKEN=your-pushover-app-token

# Optional: For local development with Ollama
# Uncomment the line below to use local LLM
# USE_LOCAL_LLM=1
```

**Where to get keys:**
- **OpenAI API Key**: Sign up at [platform.openai.com](https://platform.openai.com/), go to API Keys section
- **Pushover Keys**: Create account at [pushover.net](https://pushover.net/), create an application to get token

### Step 7: Personalize Your Chatbot

1. **Update your LinkedIn profile:**
   - Export your LinkedIn profile as PDF (LinkedIn → Me → View Profile → More → Save to PDF)
   - Replace `me/linkedin.pdf` with your PDF file

2. **Update your summary:**
   - Edit `me/summary.txt` with a brief summary about yourself, your skills, and background
   
   Example:
   ```txt
   I am a Senior Software Engineer with 10+ years of experience in AI/ML and full-stack development.
   My expertise includes Python, JavaScript, cloud architectures, and machine learning systems.
   I'm passionate about building scalable AI applications and mentoring junior developers.
   ```

3. **Update your name in the code:**
   - Open `app/app.py`
   - Find the line `self.name = "Thien Mai"` in the `Me` class `__init__` method
   - Change it to your name:
     ```python
     self.name = "Your Full Name"
     ```

## Running Locally

### Option A: With Local Ollama (Windows Host)

This setup allows your Ubuntu VM to use Ollama running on your Windows host machine. This option is perfect for testing locally before deploying, without spending a dime on a commercial LLM API.

#### On Windows Host:

1. **Install Ollama:**
   - Download from [ollama.com](https://ollama.com/)
   - Install and run

2. **Pull the required model:**
   ```powershell
   ollama pull llama3.2
   ```

3. **Configure Ollama to accept connections from VM:**
   - Set environment variable `OLLAMA_HOST=0.0.0.0:11434`
   - Restart Ollama service

4. **Configure Windows Firewall:**
   - Allow incoming connections on port 11434
   - Or add a firewall rule:
     ```powershell
     New-NetFirewallRule -DisplayName "Ollama" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow
     ```

#### On Ubuntu VM:

1. **Update your `.env` file:**
   ```env
   USE_LOCAL_LLM=1
   ```

2. **Verify connectivity** (optional):
   ```bash
   curl http://10.0.2.2:11434/api/tags
   ```
   
   Note: `10.0.2.2` is the VirtualBox host IP address when using NAT networking.

3. **Run the application:**
   ```bash
   cd app
   uv run gradio app.py
   ```

4. Open your browser to `http://127.0.0.1:7860`

**Note:** The application uses `10.0.2.2` as the host IP, which is VirtualBox's default gateway when using NAT. If you're using a different network mode (Bridged, Host-only), you'll need to update the IP in `app.py`.

### Option B: With OpenAI API

1. Ensure your `.env` has `OPENAI_API_KEY` set
2. Make sure `USE_LOCAL_LLM` is NOT set or is set to `0`
3. Run the application:
   ```bash
   cd app
   uv run gradio app.py
   ```

4. Open your browser to the URL shown (typically `http://127.0.0.1:7860`)

## Deploying to HuggingFace Spaces

### Prerequisites

1. **HuggingFace Account:**
   - Sign up at [huggingface.co](https://huggingface.co/)
   - Create an access token at Settings → Access Tokens

2. **OpenAI API Key:**
   - Required for deployment (local LLM not supported on HuggingFace)

3. **Pushover Keys (Optional):**
   - For receiving notifications about visitor interactions

### Deployment Steps

1. **Set up HuggingFace credentials:**
   ```bash
   # Login to HuggingFace
   huggingface-cli login
   # Enter your access token when prompted
   ```

2. **Ensure your `.env` is configured:**
   - Remove or comment out `USE_LOCAL_LLM=1`
   - Verify `OPENAI_API_KEY` is set

3. **Deploy with Gradio:**
   ```bash
   cd app
   uv run gradio deploy
   ```

4. **Follow the prompts:**
   - Choose a space name
   - Select visibility (public/private)
   - Confirm deployment

5. **Add environment secrets on HuggingFace:**
   - Go to your Space on HuggingFace
   - Navigate to Settings → Repository secrets
   - Add the following secrets:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `PUSHOVER_USER`: Your Pushover user key (optional)
     - `PUSHOVER_TOKEN`: Your Pushover token (optional)

6. **Access your deployed chatbot:**
   - Your space will be available at `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`

### Updating Your Deployment

After making changes to your code or personal files:

```bash
cd app
uv run gradio deploy
```

Gradio will update your existing space with the new changes.

## Project Structure

```
totem/
├── app/
│   ├── app.py              # Main application file
│   ├── README.md           # HuggingFace Space configuration
│   ├── requirements.txt    # Dependencies for HuggingFace
│   └── me/
│       ├── linkedin.pdf    # Your LinkedIn profile PDF
│       └── summary.txt     # Your personal summary
├── .env                    # Environment variables (create this)
├── pyproject.toml          # Project dependencies and metadata
├── LICENSE                 # License file
└── README.md               # This file
```

## How It Works

1. **Initialization:** The chatbot loads your LinkedIn PDF and summary text file
2. **Context Building:** It creates a system prompt that includes your background information
3. **Conversation:** Users interact with the chatbot through a Gradio interface
4. **Tool Calls:** The AI can:
   - Record user contact information when they express interest
   - Log questions it couldn't answer for future improvement
5. **Notifications:** When configured, Pushover sends you real-time notifications

## Troubleshooting

### Common Issues

**Problem:** `uv: command not found`
- **Solution:** Restart your terminal or run `source $HOME/.cargo/env`

**Problem:** Cannot connect to Ollama on Windows host
- **Solution:** 
  - Verify Ollama is running on Windows
  - Check firewall settings
  - Confirm VirtualBox network is NAT mode
  - Test connectivity: `curl http://10.0.2.2:11434/api/tags`

**Problem:** `ModuleNotFoundError` when running the app
- **Solution:** Run `uv sync` again to ensure all dependencies are installed

**Problem:** Gradio deploy fails
- **Solution:** 
  - Verify you're logged in: `huggingface-cli whoami`
  - Check your internet connection
  - Ensure `requirements.txt` is present in `app/` directory

**Problem:** OpenAI API errors
- **Solution:**
  - Verify your API key is correct in `.env`
  - Check your OpenAI account has credits
  - Ensure `.env` file is in the correct location

## Development

To modify the chatbot behavior:
- Edit system prompts in `app.py` → `system_prompt()` method
- Add new tools by creating functions and tool definitions
- Modify the chat logic in the `chat()` method

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Feel free to submit issues or pull requests to improve this project!

## Support

If you encounter any issues or have questions:
1. Check the Troubleshooting section above
2. Review the console output for error messages
3. Open an issue on GitHub with detailed information about your problem

---

**Built with:** Python, Gradio, OpenAI API, UV Package Manager
