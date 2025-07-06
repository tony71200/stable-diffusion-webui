import gradio as gr
import os
from modules import scripts, sd_models, shared, processing, sd_samplers
from modules.processing import StableDiffusionProcessingTxt2Img
from PIL import Image

class Script(scripts.Script):
    def title(self):
        return "Batch Prompt Advanced"

    def ui(self, is_img2img):
        # # Text Box show Processing and Relatived Information
        # process_box = gr.Textbox(label="Process Infor", interactive=False, lines=5)
        # Default prompt
        default_positive = gr.Textbox(label="Default Positive Prompt", lines=2, value="masterpiece, best quality")
        default_negative = gr.Textbox(label="Default Negative Prompt", lines=2, value="lowres, bad anatomy")

        # Prompt file (drag and drop or click)
        prompt_file = gr.File(label="Prompt .txt file", file_types=[".txt"])
        
        # Checkpoints dropdown
        ckpt_names = list(sd_models.checkpoints_list.keys())
        checkpoint = gr.Dropdown(label="Checkpoints", choices=ckpt_names, multiselect=True, value=ckpt_names[:1])

        # Samplers dropdown
        sampler_names = [sampler.name for sampler in sd_samplers.all_samplers]
        samplers = gr.Dropdown(label="Samplers", choices=sampler_names, multiselect=True, value=[sampler_names[0]])

        
        # Delimiter
        delimiter = gr.Textbox(label="Delimiter (e.g. ###)", value="###")
        with gr.Group():
            enable_face_restore = gr.Checkbox(label="Enable Face Restore Option", value=False)
            with gr.Column(visible=False) as face_model_group:
                face_model_dropdown = gr.Dropdown(label="Face Restore Model", choices=["CodeFormer", "GFPGAN"], value="CodeFormer")
        def toggle_face_model_group(enable):
            return gr.update(visible=enable)
        
        enable_face_restore.change(fn=toggle_face_model_group, inputs=[enable_face_restore], outputs=[face_model_group])

        return [default_positive, default_negative, prompt_file, checkpoint, samplers, delimiter, enable_face_restore, face_model_dropdown]

    def run(self, p: StableDiffusionProcessingTxt2Img, default_pos, default_neg, prompt_file, ckpt_list, sampler_list, delimiter, enable_face_restore, face_model_dropdown):
        all_outputs = []
        log_lines = []
        delimiter = delimiter.strip()

        # Read prompt file
        if prompt_file is None:
            print("[!] No prompt file provided.")
            return p  # return nothing processed

        lines = prompt_file.name if hasattr(prompt_file, 'name') else prompt_file
        with open(lines, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        last_process = None
        total_job = len(ckpt_list) * len(sampler_list) * len(prompts)
        

        print(f"[BatchPrompt] Total Images: {total_job}")

        initial_width = p.width
        initial_height = p.height
        count_image = 0
        for ckpt in ckpt_list:
            # ckpt_info = None
            # for name, info in sd_models.checkpoints_list.items():
            #     if name.strip() == ckpt.strip() or info.title.strip() == ckpt.strip():
            #         ckpt_info = info
            #         break
            ckpt_info = next((info for name, info in sd_models.checkpoints_list.items() if name.strip() == ckpt.strip() or info.title.strip() == ckpt.strip()), None)

            if not ckpt_info:
                print(f"[!] Checkpoint not found: {ckpt}")
                continue

            print(f"🔁 Loading checkpoint: {ckpt_info.title}")
            sd_models.reload_model_weights(shared.sd_model, ckpt_info)

            for index_prompt, line in enumerate(prompts, start=1):
            # for sampler in sampler_list:
            #     p.sampler_name = sampler
                
                # for line in prompts:
                for sampler in sampler_list:
                    back_to_initial_size = False
                    p.sampler_name = sampler
                    if delimiter in line:
                        parts = line.split(delimiter)
                        pos = parts[0].strip()
                        neg = parts[1].strip() if len(parts) > 1 else ""
                        size = parts[2].strip() if len(parts) > 2 else ""
                    else:
                        pos, neg, size = line.strip(), "", ""

                    full_positive = f"{default_pos}, {pos}".strip(", ")
                    full_negative = f"{default_neg}, {neg}".strip(", ")

                    p.prompt = full_positive
                    p.negative_prompt = full_negative
                    p.seed = -1  # random seed
                    p.sampler_name = sampler

                    if size and 'x' in size.lower():
                        try:
                            w, h = map(int, size.lower().split('x'))
                            p.width = w
                            p.height = h
                            back_to_initial_size = True
                        except:
                            print(f"[!] Invalid size format: {size}, using default.")

                    if enable_face_restore:
                        p.restore_faces = True
                        shared.opts.face_restoration_model = face_model_dropdown
                    else:
                        p.restore_faces = False
                    

                    print(f"🧠 Prompt: {p.prompt}")
                    print(f"🚫 Negative: {p.negative_prompt}")
                    print(f"Size: {p.width}x{p.height}")
                    face_restore_string = f"True with {face_model_dropdown}" if enable_face_restore else "False"
                    print(f"📦 Checkpoint: {ckpt_info.title}, 🎚 Sampler: {sampler}, Face Restore: {face_restore_string} ")
                    print(f"📦 Processed: {count_image + 1} / {total_job} with line number: {index_prompt}/ {len(prompts)}")

                    status_string = (f"📦 Processed: {count_image + 1} / {total_job} with line number: {index_prompt}\n \
                                       🧠 Prompt: {p.prompt} \n \
                                       Size: {p.width}x{p.height} \n \
                                       📦 Checkpoint: {ckpt_info.title}, 🎚 Sampler: {sampler}, Schedule type: {p.scheduler}")

                    proc = processing.process_images(p)
                    all_outputs.extend(proc.images)
                    last_process = proc

                    seed_used = proc.seed if hasattr(proc, "seed") else "N/A"
                    log_lines.append(f"Checkpoint: {ckpt_info.title} | Sampler: {sampler} | Seed: {seed_used}\nPositive: {p.prompt}\nNegative: {p.negative_prompt}\n")
                    
                    if back_to_initial_size:
                        p.width = initial_width
                        p.height = initial_height

                    count_image+=1
                    

        # Log to file
        log_path = os.path.join(shared.opts.outdir_txt2img_samples, "batch_prompt_log.txt")
        with open(log_path, "w+", encoding="utf-8") as f:
            f.write("\n\n".join(log_lines))

        print(f"\n✅ Batch complete. Log saved to: {log_path}")
        status_string = (f"✅ DONE! {total_job} image generated. Log save to: {log_path}")
        

        return last_process
