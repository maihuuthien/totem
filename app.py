"""Main application code for the Totem Chat + TTS app"""

import json
import sys
import os
import tempfile

from datetime import date
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs

import requests
import httpx

from pypdf import PdfReader
import gradio as gr

# Monkey patch
from huggingface_hub import hf_hub_download
import huggingface_hub
def cached_download(*args, **kwargs):
    """A simple wrapper around hf_hub_download to monkey patch cached_download"""
    return hf_hub_download(*args, **kwargs)
huggingface_hub.cached_download = cached_download

# Set up environment variables and paths
os.environ["TOTEM_APP_DIR"] = os.path.dirname(os.path.abspath(__file__))
os.environ["LATENTSYNC_DIR"] = os.path.join(os.environ["TOTEM_APP_DIR"], "LatentSync/")
sys.path.append(os.environ["LATENTSYNC_DIR"])

os.environ["CHECKPOINTS_DIR"] = os.path.join(os.environ["LATENTSYNC_DIR"], "checkpoints/")
os.makedirs(os.environ["CHECKPOINTS_DIR"], exist_ok=True)

import spaces  # pylint: disable=wrong-import-position,unused-import
from omegaconf import OmegaConf  # pylint: disable=wrong-import-position
from LatentSync.scripts.inference import prepare_for_pipeline, run_pipeline  # pylint: disable=wrong-import-position


load_dotenv(override=True)

def push(text):
    """Send a push notification using Pushover"""
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        },
        timeout=5,
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    """Record user details for follow-up"""
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    """Record a question that couldn't be answered"""
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:
    """Class representing myself with chat and TTS capabilities"""

    def __init__(self):
        self.init_openai()
        self.init_elevenlabs()
        self.init_latentsync()
        self.init_my_context()

    def init_openai(self):
        """Initialize OpenAI client"""
        print("Initializing OpenAI client...", flush=True)
        if int(os.getenv("USE_LOCAL_LLM", '0')):
            self.openai = OpenAI(
                base_url='http://10.0.2.2:11434/v1', api_key='ollama',
                http_client=httpx.Client(
                    transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                )
            )
            self.model_name = "llama3.2"
        else:
            self.openai = OpenAI()
            self.model_name = "gpt-4o-mini"

    def init_elevenlabs(self):
        """Initialize ElevenLabs client"""
        print("Initializing ElevenLabs client...", flush=True)
        self.elevenlabs = ElevenLabs(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        )

    def init_latentsync(self):
        """Initialize LatentSync pipeline for text-to-video synthesis"""

        if not int(os.getenv("USE_DOCKER_SDK", '0')):
            print("Downloading LatentSync checkpoints...", flush=True)
            hf_hub_download(
                repo_id="ByteDance/LatentSync",
                filename="whisper/tiny.pt",
                local_dir=os.environ["CHECKPOINTS_DIR"],
                force_download=False,
            )
            hf_hub_download(
                repo_id="ByteDance/LatentSync",
                filename="latentsync_unet.pt",
                local_dir=os.environ["CHECKPOINTS_DIR"],
                force_download=False,
            )

        print("Preparing LatentSync pipeline...", flush=True)
        self.unet_config = OmegaConf.load(os.path.join(
            os.environ["LATENTSYNC_DIR"], "configs/unet/second_stage.yaml"
        ))
        self.weight_dtype, self.audio_model_path, self.unet, self.scheduler = prepare_for_pipeline(
            unet_config=self.unet_config, inference_ckpt_path=os.path.join(
                os.environ["LATENTSYNC_DIR"], "checkpoints/latentsync_unet.pt"
            ),
        )

        # # NOTE: For testing purpose only
        # video_out_path = os.path.join(
        #     os.environ["TOTEM_APP_DIR"], "out_video.mp4"
        # )
        # run_pipeline(
        #     audio_model_path=self.audio_model_path,
        #     unet_config=self.unet_config,
        #     unet=self.unet,
        #     scheduler=self.scheduler,
        #     video_path=os.path.join(
        #         os.environ["TOTEM_APP_DIR"], "me/ref_video.mp4"
        #     ),
        #     audio_path=os.path.join(
        #         os.environ["TOTEM_APP_DIR"], "me/ref_audio.mp3"
        #     ),
        #     video_out_path=video_out_path,
        #     num_inference_steps=20//2,
        #     guidance_scale=1.5,
        #     weight_dtype=self.weight_dtype,
        #     width=self.unet_config.data.resolution//2,
        #     height=self.unet_config.data.resolution//2,
        # )

    def init_my_context(self):
        """Initialize personal context from files"""
        print("Loading personal context from files...", flush=True)
        self.name = "Thien Mai"

        self.linkedin = ""
        reader = PdfReader("me/linkedin.pdf")
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        self.summary = ""
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        """Handle tool calls from the assistant"""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        """Generate system prompt with personal context"""

        today = date.today()
        formatted_date = today.strftime("%B %d, %Y")

        system_prompt = f"Today is {formatted_date}. You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    def chat(self, message, history):
        """Generate assistant response based on user message and chat history"""
        messages = [
            {"role": "system", "content": self.system_prompt()}
        ] + history + [
            {"role": "user", "content": message}
        ]
        while True:
            response = self.openai.chat.completions.create(
                model=self.model_name, messages=messages, tools=tools
            )
            if response.choices[0].finish_reason != "tool_calls":
                break

            message = response.choices[0].message
            tool_calls = message.tool_calls
            results = self.handle_tool_call(tool_calls)
            messages.append(message)
            messages.extend(results)

        return response.choices[0].message.content

    def text_to_speech(self, text):
        """Convert text to speech, write to a temp MP3 file, and return its path"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            try:
                response = self.elevenlabs.text_to_speech.convert(
                    text=text,
                    voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
                    model_id="eleven_turbo_v2_5",
                    output_format="mp3_44100_128",
                )
                with open(tmp.name, "wb") as f:
                    for chunk in response:
                        if chunk:
                            f.write(chunk)

                return tmp.name

            except Exception as e:  # pylint: disable=broad-except
                print(f"ElevenLabs Error: {e}.\nFallback to OpenAI text-to-speech", flush=True)
                try:
                    with self.openai.audio.speech.with_streaming_response.create(
                        model="tts-1",
                        voice="fable",
                        input=text,
                        response_format="mp3",
                    ) as response:
                        response.stream_to_file(tmp.name)

                    return tmp.name

                except Exception as exc:  # pylint: disable=broad-except
                    print(f"OpenAI Error: {exc}", flush=True)

        return None

    def text_to_video(self, text):
        """Convert text to video, write to a temp MP4 file, and return its path"""
        audio_path = self.text_to_speech(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            try:
                run_pipeline(
                    audio_model_path=self.audio_model_path,
                    unet_config=self.unet_config,
                    unet=self.unet,
                    scheduler=self.scheduler,
                    video_path=os.path.join(
                        os.environ["TOTEM_APP_DIR"], "me/ref_video_42s.mp4"
                    ),
                    audio_path=audio_path,
                    video_out_path=tmp.name,
                    num_inference_steps=20,
                    guidance_scale=1.5,
                    weight_dtype=self.weight_dtype,
                    width=self.unet_config.data.resolution,
                    height=self.unet_config.data.resolution,
                )

                return tmp.name

            except Exception as e:  # pylint: disable=broad-except
                print(f"LatentSync Error: {e}.\n", flush=True)

        return None


if __name__ == "__main__":
    me = Me()

    with gr.Blocks(title="Totem Chat + TTS") as demo:
        gr.Markdown("# Chat with Thien Mai\nPlease log in HuggingFace to play the last answer with my avatar and voice clone.")

        video_out = gr.Video(autoplay=True)

        try:
            chatbot = gr.Chatbot(type="messages", height=100)
        except TypeError:
            chatbot = gr.Chatbot(height=100)

        with gr.Row():
            txt = gr.Textbox(placeholder="Type your message and press Enter")
            play_btn = gr.Button("Play last answer")

        last_answer = gr.State("")

        def respond(user_message, history):
            """Generate assistant response and update chat history"""
            assistant_reply = me.chat(user_message, history or [])
            # Update chat history
            history = (history or []) + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_reply}
            ]
            return history, "", assistant_reply

        txt.submit(respond, inputs=[txt, chatbot], outputs=[chatbot, txt, last_answer])
        play_btn.click(me.text_to_video, inputs=[last_answer], outputs=[video_out])

    demo.launch()
