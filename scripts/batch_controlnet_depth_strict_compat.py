import os
import sys
from PIL import Image, PngImagePlugin
import gradio as gr
from modules import scripts, sd_models, shared, processing, sd_samplers
from modules.processing import StableDiffusionProcessingTxt2Img
import numpy as np
import math

# Add path to ControlNet internal modules
controlnet_path = os.path.join(os.path.dirname(__file__), "../extensions/sd-webui-controlnet")
sys.path.append(controlnet_path)
from internal_controlnet.args import ControlNetUnit


class Script(scripts.Script):
    def title(self):
        return "Batch ControlNet Depth"

    def ui(self, is_img2img):
        with gr.Tab("🖼 Prompt Source & Images"):
            default_positive = gr.Textbox(label="Default Positive Prompt", lines=2, value="masterpiece, best quality")
            default_negative = gr.Textbox(label="Default Negative Prompt", lines=2, value="lowres, bad anatomy")

            with gr.Row(equal_height=True):
                depth_folder = gr.Textbox(label="Path to Depth Images Folder", interactive=False, scale=10)
                upload_button = gr.UploadButton("📂", file_count="directory", scale=1)

            gallery = gr.Gallery(label="Depth Images Preview", columns=4, height="auto")
            # Hidden component to store file list for the 'run' method
            gallery_files_for_run = gr.State([])

            def on_upload_dir(files):
                if not files:
                    return gr.update(), gr.update(), gr.update()
                
                folder_path = os.path.dirname(files[0].name)
                image_paths = [f.name for f in files if f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                return folder_path, image_paths, image_paths

            upload_button.upload(fn=on_upload_dir, inputs=[upload_button], outputs=[depth_folder, gallery, gallery_files_for_run], show_progress="full")

        with gr.Tab("⚙️ Generation Settings"):
            def get_ckpt_list():
                return list(sd_models.checkpoints_list.keys())

            with gr.Row():
                checkpoint = gr.Dropdown(label="Checkpoints", choices=get_ckpt_list(), multiselect=True, value=get_ckpt_list()[:1], scale=10)
                refresh_ckpt_btn = gr.Button(value="🔄", scale=1)

            def refresh_ckpt():
                # sd_models.list_models() # Refreshes the list
                new_choices = get_ckpt_list()
                return gr.update(choices=new_choices, value=new_choices[:1] if new_choices else None)

            refresh_ckpt_btn.click(fn=refresh_ckpt, inputs=[], outputs=[checkpoint])

            sampler_names = [sampler.name for sampler in sd_samplers.all_samplers]
            samplers = gr.Dropdown(label="Samplers", choices=sampler_names, multiselect=True, value=[sampler_names[0]])

            delimiter = gr.Textbox(label="Delimiter", value="###")

            controlnet_models = [os.path.splitext(f)[0] for f in os.listdir("extensions/sd-webui-controlnet/models") if "depth" in f]
            controlnet_model = gr.Dropdown(label="ControlNet Depth Model", choices=controlnet_models, value=controlnet_models[-1])
            resize_mode = gr.Dropdown(label="Resize Mode", choices=["Just Resize", "Crop and Resize", "Resize and Fill"], value="Resize and Fill")
            guidance_start = gr.Slider(label="Guidance Start", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
            guidance_end = gr.Slider(label="Guidance End", minimum=0.0, maximum=1.0, step=0.01, value=0.8)

            control_mode = gr.Dropdown(
                label="Control Mode",
                choices=["Balanced", "My prompt is more important", "ControlNet is more important"],
                value="Balanced"
            )
            pixel_perfect = gr.Checkbox(label="Pixel Perfect Mode", value=False)

            with gr.Group():
                enable_face_restore = gr.Checkbox(label="Enable Face Restore Option", value=False)
                with gr.Column(visible=False) as face_model_group:
                    face_model_dropdown = gr.Dropdown(label="Face Restore Model", choices=["CodeFormer", "GFPGAN"], value="CodeFormer")

            enable_face_restore.change(fn=lambda x: gr.update(visible=x), inputs=[enable_face_restore], outputs=[face_model_group])

        return [default_positive, default_negative, depth_folder, gallery_files_for_run,
                checkpoint, samplers, delimiter,
                controlnet_model, resize_mode, guidance_start, guidance_end,
                enable_face_restore, face_model_dropdown, control_mode, pixel_perfect]

    def run(self, p: StableDiffusionProcessingTxt2Img, default_pos, default_neg, folder_path, gallery_files,
            ckpt_list, sampler_list, delimiter,
            controlnet_model, resize_mode, guidance_start, guidance_end,
            enable_face_restore, face_model_dropdown, control_mode, pixel_perfect):

        all_outputs = []
        image_paths = gallery_files
        last_process = None
        if not image_paths:
            print("❌ No images provided.")
            return p
        
        total_job = len(ckpt_list) * len(sampler_list) * len(image_paths)
        default_p_width = p.width
        default_p_height = p.height
        count_image = 0

        for ckpt in ckpt_list:
            ckpt_info = next((info for name, info in sd_models.checkpoints_list.items() if name.strip() == ckpt.strip()), None)
            if not ckpt_info:
                print(f"[!] Checkpoint not found: {ckpt}")
                continue

            sd_models.reload_model_weights(shared.sd_model, ckpt_info)
            for index, img_path in enumerate(image_paths):
                for sampler in sampler_list:
                    p.sampler_name = sampler

                
                    try:
                        image = Image.open(img_path)
                        if image.mode != "RGB":
                            image = image.convert("RGB")

                        width, height = image.size
                        # image = image.resize((512, 512), Image.BILINEAR)
                        
                        def round_to_32(x):
                            return int(math.ceil(x / 32.0)) * 32

                        if height > width:
                            new_height = default_p_height
                            new_width = int(width * (new_height / height))
                        else: # width >= height
                            new_width = default_p_height # Assuming square-like output based on default height
                            new_height = int(height * (new_width / width))

                        # p.width = round_to_32(new_width)
                        # p.height = round_to_32(new_height)
                        p.width = new_width
                        p.height = new_height

                        # control_image = image.resize((p.width, p.height), Image.LANCZOS)

                        meta: PngImagePlugin.PngInfo = image.info
                        param_line = meta.get("parameters", "")
                        if delimiter in param_line:
                            parts = param_line.split(delimiter)
                            pos = parts[0].strip()
                            neg = parts[1].strip() if len(parts) > 1 else ""
                        else:
                            pos, neg = param_line.strip(), ""

                        full_positive = f"{default_pos}, {pos}".strip(", ")
                        full_negative = f"{default_neg}, {neg}".strip(", ")

                        p.prompt = full_positive
                        p.negative_prompt = full_negative
                        p.seed = -1

                        if enable_face_restore:
                            p.restore_faces = True
                            shared.opts.face_restoration_model = face_model_dropdown
                        else:
                            p.restore_faces = False
                        image = np.array(image)
                        unit = ControlNetUnit(
                            enabled=True,
                            module='none',
                            model=controlnet_model,
                            weight=0.95,
                            guidance_start=guidance_start,
                            guidance_end=guidance_end,
                            resize_mode=resize_mode,
                            control_mode=control_mode,
                            pixel_perfect=pixel_perfect,
                            image=image
                        )

                        # Inject the ControlNetUnit into the script arguments
                        cn_script = None
                        for script in scripts.scripts_txt2img.alwayson_scripts:
                            if script.title().lower() == "controlnet":
                                cn_script = script
                                break
                        
                        if cn_script:
                            script_args = list(p.script_args)
                            script_args[cn_script.args_from] = unit
                            p.script_args = tuple(script_args)
                        else:
                            print("❌ ControlNet script not found. Make sure it's enabled in the UI.")


                        print(f"🧠 Prompt: {p.prompt}")
                        print(f"🚫 Negative: {p.negative_prompt}")
                        print(f"📏 Size: {p.width}x{p.height}")
                        face_restore_string = f"True with {face_model_dropdown}" if enable_face_restore else "False"
                        print(f"📦 Checkpoint: {ckpt_info.title}, 🎚 Sampler: {sampler}, Face Restore: {face_restore_string} ")
                        print(f"🖼️ Processed: {count_image + 1} / {total_job} with image: {os.path.basename(img_path)}")
                        proc = processing.process_images(p)
                        all_outputs.extend(proc.images)
                        last_process = proc

                    except Exception as e:
                        print(f"❌ Failed to process {os.path.basename(img_path)}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    finally:
                        count_image += 1

        # Restore original p object state
        p.width = default_p_width
        p.height = default_p_height
        
        return last_process
    
