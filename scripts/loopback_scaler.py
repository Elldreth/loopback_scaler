import numpy as np
import math
import modules.scripts as scripts
import gradio as gr
import time
from modules import processing, images
from modules.processing import Processed
from modules.shared import opts, state

# Import PIL libraries
from PIL import ImageFilter, ImageEnhance

# This is a modification of the Loopback script. Thank you to the original author for making this available.
# This modification came from a process that I learned from the AI community to improve details and prepare an
# image for post-processing.

class Script(scripts.Script):
    def title(self):
        return "Loopback Scaler"

    def show(self, is_img2img):
        return is_img2img
    help_text = "<strong>Loops:</strong> The number of times the script will inference your image and increase the resolution in increments. The amount the resolution is increased each loop is determined by this number and the maximum image width/height.  The more loops, the more chances of your image picking up more detail, but also artifacts.  4 to 10 is what I find to work best, but you may like more or less.<br><br><strong>Denoise change:</strong> This setting will increase or decrease the denoising strength every loop.  A higher value will increase the denoising strength, while a lower value will decrease it. A setting of 1 keeps the denoising strength as it is set on the img2img settings.<br><br><strong>Dimension change:</strong> This setting changes the amount of resolution increase or decrease per loop, keeping the changes from being linear. You will get non-linear increases in image size based on which easing option you choose.  To increase the image size earlier in the process, choose one of the 'Ease Out' options, to increase the image size later in the process, choose an 'Ease In' option, to place image increases more toward the center of the process, use an 'Ease InOut' option.<br><br><strong>Maximum Image Width/Height:</strong> These parameters set the maximum width and height of the final image. Always start with an image smaller than these dimensions.  The smaller you start, the more impressive the results. I usually start at either 340x512 or 512x768<br><br><strong>Detail, Blur, Smooth, Contour:</strong> These parameters are checkboxes that apply a PIL Image Filter to the final image.<br><br><strong>Sharpness, Brightness, Color, Contrast:</strong> These parameters are sliders that adjust the sharpness, brightness, color, and contrast of the image. 1 will result in no adjustments, less than one reduces these settings for the final image and greater than 1 increases these settings.<br><br><strong>Img2Img Settings:</strong>  I recommend creating an image with txt2img and then sending the result to img2img with the prompt and settings.  For this script I use these settings..<br><br><strong>Resize mode -</strong> Crop and resize<br><strong>Sampling method -</strong> DDIM<br><strong>Sampling steps -</strong> 30<br><strong>Width/Height -</strong> 340x512 or 512x768.  I’d try to keep to the aspect ratio of the original image but these can be set lower than the resolution of the original image<br><strong>CFG Scale -</strong> 6 to 8<br><strong>Denoising strength -</strong> 0.2 to 0.4 is usual.  The lower you go, the less change between loops.  The higher you go the less the end result will look like the original image.<br><strong>Seed -</strong> This doesn’t matter too much, I usually keep it at -1</p>"
    detail_choices = ["None", "Low", "Medium", "High"]
    dim_increase_options = ["Linear",
                            "Ease In: Sine",
                            "Ease In: Cubic",
                            "Ease In: Quint",
                            "Ease In: Circ",
                            "Ease Out: Sine",
                            "Ease Out: Cubic",
                            "Ease Out: Quint",
                            "Ease Out: Circ",
                            "Ease InOut: Sine",
                            "Ease InOut: Cubic",
                            "Ease InOut: Quint",
                            "Ease InOut: Circ"]
    def ui(self, is_img2img):
        with gr.Blocks():
            with gr.Box():
                with gr.Row():
                    dimension_increment_factor = gr.Dropdown(label='Dimension Increase:', choices=self.dim_increase_options, value="Linear", elem_id=self.elem_id("dimension_increment_factor"))
                with gr.Row():
                    loops = gr.Slider(minimum=1, maximum=32, step=1, label='Loops:', value=4, elem_id=self.elem_id("loops"))
                with gr.Row():
                    denoising_strength_change_factor = gr.Slider(minimum=0.9, maximum=1.1, step=0.01, label='Denoise Change:', value=1, elem_id=self.elem_id("denoising_strength_change_factor"))            
            with gr.Box():
                with gr.Row():
                    max_width = gr.Slider(minimum=512, maximum=4096, step=64, label='Maximum Image Width:', value=1024, elem_id=self.elem_id("max_width"))
                    max_height = gr.Slider(minimum=512, maximum=4096, step=64, label='Maximum Image Height:', value=1024, elem_id=self.elem_id("max_height"))
                with gr.Row():
                    use_scale =  gr.Checkbox(label='Use Scale', value=False, elem_id=self.elem_id("use_scale"))
                    scale = gr.Slider(minimum=.5, maximum=4, step=.1, label='Scale Final Image:', value=1, elem_id=self.elem_id("final_image_scale"))
            with gr.Accordion("Final Image Filters"):
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
        return [helpinfo, loops, denoising_strength_change_factor, max_width, max_height, scale, use_scale, detail_strength, blur_strength, contour_bool, smooth_strength, sharpness_strength, brightness_strength, color_strength, contrast_strength, dimension_increment_factor]

    def __get_width_from_ratio(self, height, ratio):
        new_width = math.floor(height / ratio)
        return new_width

    def __get_height_from_ratio(self, width, ratio):
        new_height = math.floor(width * ratio)
        return new_height
    
    def __get_strength_iterations(self, strength):
        if strength == "None": return 0
        elif strength == "Low": return 1
        elif strength == "Medium": return 2
        elif strength == "High": return 3
        return 0
    
    def __get_dimension_increment(self, option, perc):
        if option == "Linear": return perc
        elif option == "Ease In: Sine": return 1 - math.cos((perc * math.pi)/2)
        elif option == "Ease In: Cubic": return perc * perc * perc
        elif option == "Ease In: Quint": return perc * perc * perc * perc
        elif option == "Ease In: Circ": return 1 - math.sqrt(1 - math.pow(perc, 2))
        elif option == "Ease Out: Sine": return math.sin((perc * math.pi) / 2)
        elif option == "Ease Out: Cubic": return 1 - pow(1 - perc, 3)
        elif option == "Ease Out: Quint": return 1 - pow(1 - perc, 5)
        elif option == "Ease Out: Circ": return math.sqrt(1 - math.pow(perc - 1, 2))
        elif option == "Ease InOut: Sine": return -(math.cos(math.pi * perc) - 1) / 2
        elif option == "Ease InOut: Cubic": return 4 * perc * perc * perc if perc < 0.5 else 1 - math.pow(-2 * perc + 2, 3) / 2
        elif option == "Ease InOut: Quint": return 16 * perc * perc * perc * perc * perc if perc < 0.5 else 1 - math.pow(-2 * perc + 2, 5) / 2
        elif option == "Ease InOut: Circ": return (1 - math.sqrt(1 - math.pow(2 * perc, 2))) / 2 if perc < 0.5 else (math.sqrt(1 - math.pow(-2 * perc + 2, 2)) + 1) / 2
        return perc
    
    def __resize_to_nearest_multiple_of_m(self, width, height, m=8):
        aspect_ratio = width / height
        if width < height:
            new_width = math.ceil(width / m) * m
            new_height = round(new_width / aspect_ratio)
            new_height = math.ceil(new_height / m) * m
        else:
            new_height = math.ceil(height / m) * m
            new_width = round(new_height * aspect_ratio)
            new_width = math.ceil(new_width / m) * m
        
        return int(new_width), int(new_height)

    def run(self, p, _, loops, denoising_strength_change_factor, max_width, max_height, scale, use_scale, detail_strength, blur_strength, contour_bool, smooth_strength, sharpness_strength, brightness_strength, color_strength, contrast_strength, dimension_increment_factor):
        start_time = time.time()
        processing.fix_seed(p)
        batch_count = p.n_iter
        p.extra_generation_params = {
            "Denoising strength change factor": denoising_strength_change_factor,
            "Dimension increment factor": dimension_increment_factor,
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

        initial_seed = None
        initial_info = None

        all_images = []
        original_init_image = p.init_images
        original_prompt = p.prompt
        state.job_count = loops * batch_count
       
        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]

        #determine oritinal image h/w ratio and max h/w ratio
        base_ratio = p.height / p.width
        
        final_height = math.floor(p.height * scale) if use_scale else max_height
        final_width = math.floor(p.width * scale) if use_scale else max_width
        
        orig_height_diff = final_height - p.height
        orig_width_diff = final_width - p.width
        
        orig_height = p.height
        orig_width = p.width
        
        max_ratio = final_height / final_width
        use_height = base_ratio >= max_ratio
                    
        print("Starting Loopback Scaler")
        print(f"Original size:    {p.width}x{p.height}")
        print(f"Final size:       {final_width}x{final_height}")
        print(f"Denoising:        {denoising_strength_change_factor}")
        print(f"Dimension change: {dimension_increment_factor}")
        
        for n in range(batch_count):
            history = []

            # Reset to original init image at the start of each batch
            p.init_images = original_init_image

            for i in range(loops):
                p.n_iter = 1
                p.batch_size = 1
                p.do_not_save_grid = True
                loop_fraction = i/loops
                easing_factor = self.__get_dimension_increment(dimension_increment_factor, loop_fraction)
                
                last_image = i == loops - 1
                
                calc_height = final_height if last_image else (int((orig_height_diff * easing_factor) + orig_height ))
                calc_width = final_width if last_image else (int((orig_width_diff * easing_factor) + orig_width))

                if use_height:
                    p.width, p.height = self.__resize_to_nearest_multiple_of_m(width=self.__get_width_from_ratio(calc_height, base_ratio), height=calc_height)
                else:
                    p.width, p.height = self.__resize_to_nearest_multiple_of_m(width=calc_width, height=self.__get_height_from_ratio(calc_height, base_ratio))
                
                print()
                print(f"Loopback Scaler:    {i+1}/{loops}")
                print(f"Iteration size:     {p.width}x{p.height}")
                print(f"Denoising strength: {p.denoising_strength}")

                if opts.img2img_color_correction:
                    p.color_corrections = initial_color_corrections

                state.job = f"Iteration {i + 1}/{loops}, batch {n + 1}/{batch_count}"
                
                # Processing image
                processed = processing.process_images(p)
                
                if last_image:        
                    processed.images[0] = ImageEnhance.Sharpness(processed.images[0]).enhance(sharpness_strength)
                    processed.images[0] = ImageEnhance.Brightness(processed.images[0]).enhance(brightness_strength)
                    processed.images[0] = ImageEnhance.Color(processed.images[0]).enhance(color_strength)
                    processed.images[0] = ImageEnhance.Contrast(processed.images[0]).enhance(contrast_strength)
                    
                    for j in range(self.__get_strength_iterations(detail_strength)):
                        processed.images[0] = processed.images[0].filter(ImageFilter.DETAIL)

                    for j in range(self.__get_strength_iterations(smooth_strength)):
                        processed.images[0] = processed.images[0].filter(ImageFilter.SMOOTH)

                    for j in range(self.__get_strength_iterations(blur_strength)):
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
            
            all_images += history

        end_time = time.time()
        print("Loopback Scaler: All Done!")
        print(f"LS: {round(end_time - start_time)}s elapsed")
        print()
        processed = Processed(p, all_images, p.all_seeds, initial_info,)
        
        return processed