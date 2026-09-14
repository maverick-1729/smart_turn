import sys
from pathlib import Path

import gradio as gr
import spaces

sys.path.insert(0, str(Path(__file__).parent / "src" / "smart_turn"))

from inference import DEFAULT_CHECKPOINT, predict

@spaces.GPU
def process_recording(audio):
    """
    Runs inference over the completed microphone recording.  Gradio provides
    this as a ``(sample_rate, samples)`` tuple because the Audio component is
    configured with ``type="numpy"``.
    """
    if audio is None:
        return None
    return round(predict(audio, checkpoint_path=DEFAULT_CHECKPOINT), 4)


with gr.Blocks(title="Smart Turn - Hinglish Endpoint Detection") as demo:
    gr.Markdown(
        "# Smart Turn Detector (Hinglish)\n"
        "Record yourself speaking, then click **Stop Recording**. The score below "
        "shows how likely the completed recording ends with a finished turn."
    )

    audio_input = gr.Audio(
        sources=["microphone"],
        type="numpy",
        label="Live microphone input",
    )
    probability_output = gr.Number(label="P(turn finished)", precision=4)

    audio_input.stop_recording(
        fn=process_recording,
        inputs=audio_input,
        outputs=probability_output,
    )


if __name__ == "__main__":
    demo.launch()
