import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from django.shortcuts import render, redirect
from PIL import Image, ImageDraw
from django.contrib import messages
from django.contrib.staticfiles.storage import staticfiles_storage
from .models import Imag, Point, SegmentedImages, Suggestions
from mmseg.apis import inference_model, init_model, show_result_pyplot
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Create your views here.

# model initiation

## pspnet
# config_file = "main/static/mmsegmentation/configs/pspnet/pspnet_r101-d8_4xb4-80k_pascal-context-59-480x480.py"
# ckpt_path = "main/static/checkpoints/pspnet_r101-d8_480x480_80k_pascal_context_59_20210416_114418-fa6caaa2.pth"
# model = init_model(config=config_file, checkpoint=ckpt_path, device="cpu")


## sam
print("Loading SAM model...")
sam_checkpoint = "main/static/checkpoints/sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
print("Loaded SAM")

COLOR_THEMES = [
["eec9d2", "f4b6c2", "f6abb6"],
["011f4b", "005b96", "b3cde0"],
["3da4ab", "f6cd61", "fe8a71"],
["adcbe3", "e7eff6", "63ace5"],
["fec8c1", "fad9c1", "f9caa7"],
["009688", "35a79c", "54b2a9"],
["fdf498", "7bc043", "0392cf"],
["ee4035", "f37736", "fdf498"],
["ffffff", "d0e1f9", "4d648d"],
["eeeeee", "dddddd", "cccccc"],
["ff6f69", "ffcc5c", "88d8b0"],
["008744", "0057e7", "ffa700"]
]


# PASCAL Context 
CLASSES = ['background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench',
               'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
               'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
               'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
               'floor', 'flower', 'food', 'grass', 'ground', 'horse',
               'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
               'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep',
               'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table',
               'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water',
               'window', 'wood']
wall_indx = CLASSES.index("wall") - 1

def home(request):
    if request.method=="POST":
        img=Imag() 
        img.image=request.FILES.get('roomimage') 
        img.save()
        sPoints = request.POST.get('coords').split('|')
        coords = [[int(j) for j in i.split(',')] for i in sPoints if len(i) != 0]
        print(coords)
        for i in coords:
            point=Point()
            point.img_id=img
            point.x=i[0]
            point.y=i[1]
            point.save()
        return redirect('segment', pk=img.img_id)
    return render(request, 'main/home.html')

def decor(request, pk):
    img=Imag.objects.get(img_id=pk)
    if request.method == "POST":
        color=str(request.POST.get('color'))
        final=[]
        for i in (0, 2, 4):
            decimal = int(color[i+1:i+3], 16)
            final.append(decimal)
        return redirect('decor', pk=pk)
    return render(request, 'main/decor.html', {'img': img, 'pk': pk})


def segment(request, pk):
    img=Imag.objects.get(img_id=pk)
    img_rel_path = img.image.url    
    img_abs_path = f".{img_rel_path}"
    
    result = inference_model(model, img_abs_path)
    mask = result.pred_sem_seg.data
    
    img = plt.imread(img_abs_path)
    mask = np.squeeze(mask, axis=0).numpy()
    
    mask_55 = np.zeros_like(mask)
    mask_55[mask == 55] = 1
    
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the image
    ax.imshow(img)

    # wall color
    color = (255, 0, 255)
    colors = [(0, 0, 0), color]
    cmap = mcolors.ListedColormap(colors)

    # Plot the mask on top of the image
    masked_image = cmap(mask_55)
    masked_image = masked_image[:,:,:3]
    ax.imshow(masked_image, alpha=0.5)
    ax.set_axis_off()
        
    fname = os.path.basename(img_abs_path)
    fname = fname[:-4] + f"_{color}.jpg"
    fname = os.path.join("images", "results", fname)
    plt.savefig(fname)


def sam_segment(request, pk):
    points = Point.objects.filter(img_id=pk)
    img = Imag.objects.get(img_id=pk)
    img_rel_path = img.image.url
    img_abs_path = f".{img_rel_path}"
    check = SegmentedImages.objects.filter(segImg_id=img).exists()
    color_check = False

    if request.method == "POST":
        # (This part for handling color selection remains the same)
        if request.POST.get('colorchecker') == "on":
            colors = request.POST.getlist('color')
            for i in range(len(points)):
                points[i].color = colors[i]
                points[i].save()
        else:
            colors = request.POST.getlist('col1')
            for i in range(len(points)):
                if colors[i]:
                    temp_color = colors[i].split(',')
                    if len(temp_color) == 3:
                        rgb_tuple = (int(temp_color[0]), int(temp_color[1]), int(temp_color[2]))
                        points[i].color = '#%02x%02x%02x' % rgb_tuple
                        points[i].save()
        
        try:
            SegmentedImages.objects.get(segImg_id=img).delete()
        except SegmentedImages.DoesNotExist:
            pass
        check = False
        color_check = True

    colorMap = Point.objects.filter(img_id=pk).values_list('color', flat=True)

    if not check:
        print("Transforming image...")
        image = cv2.imread(img_abs_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        print("Image transformed.")

        w, h = image.shape[0], image.shape[1]
        
        # ==> OPTIMIZATION START <==
        # Collect all points into arrays for a single prediction call
        input_points = np.array([[(h / 700) * i.x, (w / 700) * i.y] for i in points])
        input_labels = np.ones(len(input_points))

        pil_image = Image.fromarray(image)

        # Make a single, efficient call to the model
        if len(input_points) > 0:
            print(f"Predicting {len(input_points)} points in a single batch...")
            masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )
            print("Prediction complete.")

            # Now, loop through the generated masks to apply colors
            for i, (mask, point_obj) in enumerate(zip(masks, points)):
                if not color_check:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    point_obj.color = '#%02x%02x%02x' % color
                    point_obj.save()
                else:
                    hex_color = point_obj.color.lstrip('#')
                    color = tuple(int(hex_color[j:j + 2], 16) for j in (0, 2, 4))
                
                # Create a transparent overlay for the mask
                mask_image = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(mask_image)
                
                # Convert boolean mask to an image mask that Pillow can use
                pillow_mask = Image.fromarray(mask * 255)
                
                # Draw the color onto the overlay using the mask
                draw.bitmap((0, 0), pillow_mask, fill=color)

                # Composite the colored overlay onto the main image
                pil_image.paste(mask_image, (0,0), mask_image)

        # ==> OPTIMIZATION END <==

        original_basename = os.path.basename(img_abs_path)
        base_name, extension = os.path.splitext(original_basename)
        new_filename = f"{base_name}_seg_{pk}{extension}"
        fname = os.path.join("images", "results", new_filename)
        
        print(f"Saving final image to: {fname}")
        
        segImg = SegmentedImages()
        segImg.segImg_id = img
        segImg.segmentedImage = fname
        pil_image.save(fname)
        segImg.save()

    segImg = SegmentedImages.objects.get(segImg_id=img.img_id)
    return render(request, 'main/decor.html', {'img': img, 'segImg': segImg, 'colorMap': colorMap, 'pk': pk})

def sam_segment(request, pk):
    points = Point.objects.filter(img_id=pk)
    img = Imag.objects.get(img_id=pk)
    img_rel_path = img.image.url
    img_abs_path = f".{img_rel_path}"
    check = SegmentedImages.objects.filter(segImg_id=img).exists()
    color_check = False

    if request.method == "POST":
        if request.POST.get('colorchecker') == "on":
            colors = request.POST.getlist('color')
            for i in range(len(points)):
                points[i].color = colors[i]
                points[i].save()
        else:
            colors = request.POST.getlist('col1')
            for i in range(len(points)):
                if colors[i]:
                    temp_color = colors[i].split(',')
                    if len(temp_color) == 3:
                        rgb_tuple = (int(temp_color[0]), int(temp_color[1]), int(temp_color[2]))
                        points[i].color = '#%02x%02x%02x' % rgb_tuple
                        points[i].save()

        try:
            SegmentedImages.objects.get(segImg_id=img).delete()
        except SegmentedImages.DoesNotExist:
            pass
        check = False
        color_check = True

    colorMap = Point.objects.filter(img_id=pk).values_list('color', flat=True)

    if not check:
        print("Transforming image")
        image = cv2.imread(img_abs_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        print("Transforming image done")

        w, h = image.shape[0], image.shape[1]
        p = [[(h / 700) * i.x, (w / 700) * i.y] for i in points]
        
        pil_image = Image.fromarray(image)
        color = (0, 0, 0)
        for point, j in zip(p, points):
            input_point = np.array([point])
            input_label = np.array([1])

            print("Predicting...")
            mask, scores, logits = predictor.predict(
                point_coords=input_point, point_labels=input_label, multimask_output=False,
            )
            mask = np.squeeze(mask, axis=0)
            mask_55 = np.zeros_like(mask)
            mask_55[mask == True] = 255

            if not color_check:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                j.color = '#%02x%02x%02x' % color
                j.save()
            else:
                hex_color = j.color.lstrip('#')
                color = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            
            draw = ImageDraw.Draw(pil_image)
            mask_pil = Image.fromarray(mask_55)
            draw.bitmap((0, 0), mask_pil, fill=color)

        # ==> CORRECTED FILENAME LOGIC
        original_basename = os.path.basename(img_abs_path)
        base_name, extension = os.path.splitext(original_basename)
        # Clean the color tuple string for the filename
        color_str = str(color).replace(' ', '')
        new_filename = f"{base_name}_{color_str}{extension}"
        fname = os.path.join("images", "results", new_filename)
        # <== END OF CORRECTION
        
        print(fname)
        
        segImg = SegmentedImages()
        segImg.segImg_id = img
        segImg.segmentedImage = fname
        pil_image.save(fname) # This should now work
        segImg.save()


    segImg = SegmentedImages.objects.get(segImg_id=img.img_id)
    return render(request, 'main/decor.html', {'img': img, 'segImg': segImg, 'colorMap': colorMap, 'pk': pk})


def suggestions(request, pk):
    img = Imag.objects.get(img_id=pk)
    points = Point.objects.filter(img_id=pk)

    if not points.exists():
        messages.error(request, "Cannot generate suggestions for an image with no selection points.")
        return redirect('home')

    if not Suggestions.objects.filter(sugImg_id=img).exists():
        img_rel_path = img.image.url
        img_abs_path = f".{img_rel_path}"

        print("Transforming image for suggestions...")
        image = cv2.imread(img_abs_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        
        w, h = image.shape[0], image.shape[1]
        original_basename = os.path.basename(img_abs_path)
        base_name, extension = os.path.splitext(original_basename)

        # ==> OPTIMIZATION: Get all masks in one call for suggestions <==
        input_points = np.array([[(h / 700) * i.x, (w / 700) * i.y] for i in points])
        input_labels = np.ones(len(input_points))
        all_masks = []
        if len(input_points) > 0:
            all_masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )

        # --- Loop for Color Theme Suggestions ---
        for theme_index, theme in enumerate(COLOR_THEMES):
            pil_image = Image.fromarray(image)
            for i, mask in enumerate(all_masks):
                theme_color_hex = theme[i % len(theme)]
                color = tuple(int(theme_color_hex[j:j+2], 16) for j in (0, 2, 4))
                
                mask_image = Image.new('RGBA', pil_image.size)
                draw = ImageDraw.Draw(mask_image)
                pillow_mask = Image.fromarray(mask * 255)
                draw.bitmap((0, 0), pillow_mask, fill=color)
                pil_image.paste(mask_image, (0,0), mask_image)
            
            new_filename = f"{base_name}_theme_{theme_index}{extension}"
            fname = os.path.join("images", "suggestions", new_filename)
            pil_image.save(fname)
            
            s_img = Suggestions(sugImg_id=img, sugImage=fname)
            s_img.save()

        # --- Loop for Texture Suggestions ---
        url = "main/" + staticfiles_storage.url('textures')
        textures = os.listdir(url)
        for i, texture in enumerate(textures):
            # Apply texture to a random mask from the pre-generated set
            if len(all_masks) > 0:
                pil_image = Image.fromarray(image)
                random_mask = random.choice(all_masks)
                
                texture_path = os.path.join(url, texture)
                texture_img = Image.open(texture_path).resize(pil_image.size)
                
                pillow_mask = Image.fromarray(random_mask * 255)
                pil_image = Image.composite(texture_img, pil_image, pillow_mask)
                
                new_filename = f"{base_name}_texture_{i}{extension}"
                fname = os.path.join("images", "suggestions", new_filename)
                pil_image.save(fname)
                
                s_img = Suggestions(sugImg_id=img, sugImage=fname)
                s_img.save()

    paths = Suggestions.objects.filter(sugImg_id=pk)
    return render(request, 'main/suggestions.html', {'pk': pk, 'img': img, 'paths': paths})
