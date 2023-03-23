import numpy as np
import math
import modules.scripts as scripts
import gradio as gr
from modules import processing, images
from modules.processing import Processed
from modules.shared import opts, state

# Import PIL libraries
from PIL import Image, ImageFilter, ImageEnhance

# This is a modification of the Loopback script. Thank you to the original author for making this available.
# This modification came from a process that I learned from the AI community to improve details and prepare an
# image for post-processing.

class Script(scripts.Script):
    def title(self):
        return "Loopback Scaler"

    def show(self, is_img2img):
        return is_img2img
    help_text = "<strong>Loops:</strong> The number of times the script will inference your image and increase the resolution in increments. The amount the resolution is increased each loop is determined by this number and the maximum image width/height.  The more loops, the more chances of your image picking up more detail, but also artifacts.  4 to 10 is what I find to work best, but you may like more or less.<br><br><strong>Denoise change:</strong> This setting will increase or decrease the denoising strength every loop.  A higher value will increase the denoising strength, while a lower value will decrease it. A setting of 1 keeps the denoising strength as it is set on the img2img settings.<br><br><strong>Adaptive change:</strong> This setting changes the amount of resolution increase per loop, keeping the changes from being linear.  The higher the value the more significant the resolution changes toward the end of the looping.<br><br><strong>Maximum Image Width/Height:</strong> These parameters set the maximum width and height of the final image. Always start with an image smaller than these dimensions.  The smaller you start, the more impressive the results. I usually start at either 340x512 or 512x768<br><br><strong>Detail, Blur, Smooth, Contour:</strong> These parameters are checkboxes that apply a PIL Image Filter to the final image.<br><br><strong>Sharpness, Brightness, Color, Contrast:</strong> These parameters are sliders that adjust the sharpness, brightness, color, and contrast of the image. 1 will result in no adjustments, less than one reduces these settings for the final image and greater than 1 increases these settings.<br><br><strong>Img2Img Settings:</strong>  I recommend creating an image with txt2img and then sending the result to img2img with the prompt and settings.  For this script I use these settings..<br><br><strong>Resize mode -</strong> Crop and resize<br><strong>Sampling method -</strong> DDIM<br><strong>Sampling steps -</strong> 30<br><strong>Width/Height -</strong> 340x512 or 512x768.  I’d try to keep to the aspect ratio of the original image but these can be set lower than the resolution of the original image<br><strong>CFG Scale -</strong> 6 to 8<br><strong>Denoising strength -</strong> 0.2 to 0.4 is usual.  The lower you go, the less change between loops.  The higher you go the less the end result will look like the original image.<br><strong>Seed -</strong> This doesn’t matter too much, I usually keep it at -1</p>"
    detail_choices = ["None", "Low", "Medium", "High"]
    def ui(self, is_img2img):
        with gr.Blocks():
            with gr.Box():
                with gr.Row():
                    loops = gr.Slider(minimum=1, maximum=32, step=1, label='Loops:', value=4, elem_id=self.elem_id("loops"))
                    denoising_strength_change_factor = gr.Slider(minimum=0.9, maximum=1.1, step=0.01, label='Denoise Change:', value=1, elem_id=self.elem_id("denoising_strength_change_factor"))            
                    adaptive_increment_factor = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, label='Adaptive Change:', value=1.0, elem_id=self.elem_id("adaptive_increment_factor"))
            with gr.Box():
                with gr.Row():
                    max_width = gr.Slider(minimum=512, maximum=2048, step=64, label='Maximum Image Width:', value=1024, elem_id=self.elem_id("max_width"))
                    max_height = gr.Slider(minimum=512, maximum=2048, step=64, label='Maximum Image Height:', value=1024, elem_id=self.elem_id("max_height"))
            with gr.Box():
                with gr.Row():
                    detail_strength = gr.Dropdown(label='Add Detail', choices=self.detail_choices, value="None", elem_id=self.elem_id("detail_strength"))
                    blur_strength = gr.Dropdown(label='Add Blur', choices=self.detail_choices, value="None", elem_id=self.elem_id("blur_bool"))
                    smooth_strength = gr.Dropdown(label='Smoothing', choices=self.detail_choices, value="None", elem_id=self.elem_id("smooth_strength"))
                    contour_bool = gr.Checkbox(label='Contour', value=False, elem_id=self.elem_id("contour_bool"))
            with gr.Box():
                with gr.Row():
                    sharpness_strength = gr.Slider(minimum=0.1, maximum=2.0, step=0.01, label='Sharpness:', value=1.0, elem_id=self.elem_id("sharpness_strength")) 
                    brightness_strength = gr.Slider(minimum=0.1, maximum=2.0, step=0.01, label='Brightness:', value=1.0, elem_id=self.elem_id("brightness_strength"))
                with gr.Row():
                    color_strength = gr.Slider(minimum=0.1, maximum=2.0, step=0.01, label='Color:', value=1.0, elem_id=self.elem_id("color_strength"))
                    contrast_strength = gr.Slider(minimum=0.1, maximum=2.0, step=0.01, label='Contrast:', value=1.0, elem_id=self.elem_id("contrast_strength"))
            with gr.Accordion("Info - Loopback Scaler", open=False):
                helpinfo = gr.HTML("<p style=\"margin-bottom:0.75em\">{}</p>".format(self.help_text))
        return [helpinfo, loops, denoising_strength_change_factor, max_width, max_height, detail_strength, blur_strength, contour_bool, smooth_strength, sharpness_strength, brightness_strength, color_strength, contrast_strength, adaptive_increment_factor]

    def __get_width_from_ratio(self, height, ratio):
        new_width = math.floor(height / ratio)
        return new_width

    def __get_height_from_ratio(self, width, ratio):
        new_height = math.floor(width * ratio)
        return new_height

    def run(self, p, _, loops, denoising_strength_change_factor, max_width, max_height, detail_strength, blur_strength, contour_bool, smooth_strength, sharpness_strength, brightness_strength, color_strength, contrast_strength, adaptive_increment_factor):
        processing.fix_seed(p)
        batch_count = p.n_iter
        p.extra_generation_params = {
            "Denoising strength change factor": denoising_strength_change_factor,
            "Add Detail": detail_strength,
            "Add Blur": blur_strength,
            "Smoothing": smooth_strength,
            "Contour": contour_bool,
            "Sharpness": sharpness_strength,
            "Brightness": brightness_strength,
            "Color Strength": color_strength,
            "Contrast": contrast_strength,
        }

        p.batch_size = 1
        p.n_iter = 1

        output_images, info = None, None
        initial_seed = None
        initial_info = None

        grids = []
        all_images = []
        original_init_image = p.init_images
        original_prompt = p.prompt
        state.job_count = loops * batch_count

       
        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]

        #determine oritinal image h/w ratio and max h/w ratio
        current_ratio = p.height / p.width
        max_ratio = max_height / max_width
        use_height = False
                    
        #set loop increment to the lower of height/width and height if equal
        if current_ratio < max_ratio:
            #width will hit max first
            loop_increment = math.floor((max_width - p.width)/loops)
            use_height = False
        elif current_ratio > max_ratio:
            #height will hit max first
            loop_increment = math.floor((max_height - p.height)/loops)
            use_height = True
        else:
            #if current_ratio and max_ratio are the same, they will hit max at the same time
            loop_increment = math.floor((max_height - p.height)/loops)
            use_height = True
        
        for n in range(batch_count):
            history = []

            # Reset to original init image at the start of each batch
            p.init_images = original_init_image

            for i in range(loops):
                p.n_iter = 1
                p.batch_size = 1
                p.do_not_save_grid = True

                avg_intensity = np.mean(p.init_images[0])
                adaptive_increment = int(loop_increment * (avg_intensity / 255) * adaptive_increment_factor)
                print(f"adaptive_increment: {adaptive_increment}")

                if i < loops - 1:
                    if use_height == True:
                        newheight = p.height + adaptive_increment
                        p.height = newheight
                        p.width = self.__get_width_from_ratio(newheight, current_ratio)
                    elif use_height == False:
                        newwidth = p.width + adaptive_increment
                        p.width = newwidth
                        p.height = self.__get_height_from_ratio(newwidth, current_ratio)
                else:
                    if use_height == True:
                        p.height = max_height
                        p.width = self.__get_width_from_ratio(max_height, current_ratio)
                    else:
                        p.width = max_width
                        p.height = self.__get_height_from_ratio(max_width, current_ratio)

                if opts.img2img_color_correction:
                    p.color_corrections = initial_color_corrections

                state.job = f"Iteration {i + 1}/{loops}, batch {n + 1}/{batch_count}"
                
                processed = processing.process_images(p)
                
                if i == loops - 1:              
                    processed.images[0] = ImageEnhance.Sharpness(processed.images[0]).enhance(sharpness_strength)
                    processed.images[0] = ImageEnhance.Brightness(processed.images[0]).enhance(brightness_strength)
                    processed.images[0] = ImageEnhance.Color(processed.images[0]).enhance(color_strength)
                    processed.images[0] = ImageEnhance.Contrast(processed.images[0]).enhance(contrast_strength)
                    
                    if not detail_strength == "None":
                        if detail_strength == "Low":
                            processed.images[0] = processed.images[0].filter(ImageFilter.DETAIL)
                        elif detail_strength == "Medium":
                            for j in range(2):
                                processed.images[0] = processed.images[0].filter(ImageFilter.DETAIL)
                        elif detail_strength == "High":
                            for j in range(3):
                                processed.images[0] = processed.images[0].filter(ImageFilter.DETAIL)
                    if not smooth_strength == "None":
                        if smooth_strength == "Low":
                            processed.images[0] = processed.images[0].filter(ImageFilter.SMOOTH)
                        elif smooth_strength == "Medium":
                            for j in range(2):
                                processed.images[0] = processed.images[0].filter(ImageFilter.SMOOTH)
                        elif smooth_strength == "High":
                            for j in range(3):
                                processed.images[0] = processed.images[0].filter(ImageFilter.SMOOTH)
                    if not blur_strength == "None":
                        if blur_strength == "Low":
                            processed.images[0] = processed.images[0].filter(ImageFilter.BLUR)
                        elif blur_strength == "Medium":
                            for j in range(2):
                                processed.images[0] = processed.images[0].filter(ImageFilter.BLUR)
                        elif blur_strength == "High":
                            for j in range(3):
                                processed.images[0] = processed.images[0].filter(ImageFilter.BLUR)
                    if contour_bool == True:
                        processed.images[0] = processed.images[0].filter(ImageFilter.CONTOUR)
                    
                    images.save_image(processed.images[0], p.outpath_samples, "img2img", initial_seed, original_prompt, opts.samples_format, info=processed.info, short_filename=False,p=p)                    
                    history.append(processed.images[0])
                
                if initial_seed is None:
                    initial_seed = processed.seed
                    initial_info = processed.info

                init_img = processed.images[0]

                p.init_images = [init_img]
                p.seed = processed.seed + 1
                p.all_seeds.append(p.seed)
                p.all_subseeds.append(p.subseed)
                p.all_prompts.append(p.prompt)

                p.denoising_strength = min(max(p.denoising_strength * denoising_strength_change_factor, 0.1), 1)
                
            # grid = images.image_grid(history, rows=1)
            # if opts.grid_save:
            #     images.save_image(grid, p.outpath_grids, "grid", initial_seed, original_prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

            #grids.append(grid)
            all_images += history

        #if opts.return_grid:
        #    all_images = grids + all_images

        processed = Processed(p, all_images, p.all_seeds, initial_info,)
        
        return processed
