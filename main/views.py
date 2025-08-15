# DecorAI-main/main/views.py

import os
import cv2
import random
import numpy as np
import matplotlib.colors as mcolors
from django.shortcuts import render, redirect
from PIL import Image, ImageDraw
from django.contrib import messages
from django.contrib.staticfiles.storage import staticfiles_storage
from .models import Imag, Point, SegmentedImages, Suggestions

# ==============================================================================
# GLOBAL MODEL LOADING
# ==============================================================================
print("Initializing AI Model...")
try:
    from segment_anything import sam_model_registry, SamPredictor
    
    sam_checkpoint = "main/static/checkpoints/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    device = "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    PREDICTOR = SamPredictor(sam)
    print("AI Model loaded successfully.")
except Exception as e:
    PREDICTOR = None
    print(f"Error loading AI Model: {e}")

COLOR_THEMES = [
    ["eec9d2", "f4b6c2", "f6abb6"], ["011f4b", "005b96", "b3cde0"],
    ["3da4ab", "f6cd61", "fe8a71"], ["adcbe3", "e7eff6", "63ace5"],
    ["fec8c1", "fad9c1", "f9caa7"], ["009688", "35a79c", "54b2a9"],
    ["fdf498", "7bc043", "0392cf"], ["ee4035", "f37736", "fdf498"],
    ["ffffff", "d0e1f9", "4d648d"], ["eeeeee", "dddddd", "cccccc"],
    ["ff6f69", "ffcc5c", "88d8b0"], ["008744", "0057e7", "ffa700"]
]
# ==============================================================================

def home(request):
    if request.method == "POST":
        if not PREDICTOR:
            messages.error(request, "AI Model is not available. Please check server logs.")
            return redirect('home')
            
        img_file = request.FILES.get('roomimage')
        if not img_file:
            messages.error(request, "Please upload an image file.")
            return redirect('home')

        img = Imag(image=img_file)
        img.save()

        sPoints = request.POST.get('coords', '').split('|')
        coords = [[int(j) for j in i.split(',')] for i in sPoints if i]
        
        if not coords:
            img.delete()
            messages.error(request, "Please click on the image to select at least one area to color.")
            return redirect('home')

        points_to_create = [Point(img_id=img, x=c[0], y=c[1]) for c in coords]
        Point.objects.bulk_create(points_to_create)
        
        return redirect('segment', pk=img.img_id)
        
    return render(request, 'main/home.html')

def decor(request, pk):
    img = Imag.objects.get(img_id=pk)
    return render(request, 'main/decor.html', {'img': img, 'pk': pk})


def sam_segment(request, pk):
    points = Point.objects.filter(img_id=pk)
    img = Imag.objects.get(img_id=pk)
    img_rel_path = img.image.url
    img_abs_path = f".{img_rel_path}"
    existing_seg_img = SegmentedImages.objects.filter(segImg_id=img).first()
    color_check = False

    if request.method == "POST":
        if request.POST.get('colorchecker') == "on":
            colors = request.POST.getlist('color')
            for i, point in enumerate(points):
                point.color = colors[i]
                point.save()
        else:
            colors = request.POST.getlist('col1')
            for i, point in enumerate(points):
                if colors[i]:
                    temp_color = colors[i].split(',')
                    if len(temp_color) == 3:
                        rgb_tuple = (int(temp_color[0]), int(temp_color[1]), int(temp_color[2]))
                        point.color = '#%02x%02x%02x' % rgb_tuple
                        point.save()
        
        if existing_seg_img:
            existing_seg_img.delete()
        existing_seg_img = None
        color_check = True

    colorMap = points.values_list('color', flat=True)

    if existing_seg_img:
        segImg = existing_seg_img
    else:
        image = cv2.imread(img_abs_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        PREDICTOR.set_image(image)
        
        w, h = image.shape[0], image.shape[1]
        p = [[(h / 700) * i.x, (w / 700) * i.y] for i in points]
        pil_image = Image.fromarray(image)
        
        # Initialize color to prevent UnboundLocalError
        color = (0, 0, 0) 

        for point, point_obj in zip(p, points):
            input_point = np.array([point])
            input_label = np.array([1])
            
            mask, _, _ = PREDICTOR.predict(
                point_coords=input_point, point_labels=input_label, multimask_output=False,
            )
            mask = np.squeeze(mask, axis=0)

            if not color_check:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                point_obj.color = '#%02x%02x%02x' % color
                point_obj.save()
            else:
                hex_color = point_obj.color.lstrip('#') if point_obj.color else '000000'
                color = tuple(int(hex_color[j:j + 2], 16) for j in (0, 2, 4))

            # Use the reliable paste method to apply color
            pillow_mask = Image.fromarray(mask)
            color_fill = Image.new("RGB", pil_image.size, color)
            pil_image.paste(color_fill, mask=pillow_mask)

        original_basename = os.path.basename(img_abs_path)
        base_name, extension = os.path.splitext(original_basename)
        new_filename = f"{base_name}_seg_{pk}{extension}"
        fname = os.path.join("images", "results", new_filename)
        
        segImg = SegmentedImages(segImg_id=img, segmentedImage=fname)
        pil_image.save(fname)
        segImg.save()

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

        image = cv2.imread(img_abs_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        PREDICTOR.set_image(image)
        
        w, h = image.shape[0], image.shape[1]
        p = [[(h / 700) * i.x, (w / 700) * i.y] for i in points]
        original_basename = os.path.basename(img_abs_path)
        base_name, extension = os.path.splitext(original_basename)
        
        suggestions_to_create = []

        for theme_index, theme in enumerate(COLOR_THEMES):
            pil_image = Image.fromarray(image)
            count = 0
            for point in p:
                input_point = np.array([point])
                input_label = np.array([1])
                mask, _, _ = PREDICTOR.predict(
                    point_coords=input_point, point_labels=input_label, multimask_output=False
                )
                mask = np.squeeze(mask, axis=0)
                
                hex_color = theme[count % len(theme)]
                color = tuple(int(hex_color[j:j+2], 16) for j in (0, 2, 4))
                count += 1
                
                pillow_mask = Image.fromarray(mask)
                color_fill = Image.new("RGB", pil_image.size, color)
                pil_image.paste(color_fill, mask=pillow_mask)

            new_filename = f"{base_name}_theme_{theme_index}{extension}"
            fname = os.path.join("images", "suggestions", new_filename)
            pil_image.save(fname)
            suggestions_to_create.append(Suggestions(sugImg_id=img, sugImage=fname))

        texture_dir = "main/" + staticfiles_storage.url('textures')
        textures = os.listdir(texture_dir)
        for i, texture_name in enumerate(textures):
            if p:
                pil_image = Image.fromarray(image)
                point = random.choice(p)
                
                input_point = np.array([point])
                input_label = np.array([1])
                mask, _, _ = PREDICTOR.predict(
                    point_coords=input_point, point_labels=input_label, multimask_output=False
                )
                mask = np.squeeze(mask, axis=0)
                
                texture_path = os.path.join(texture_dir, texture_name)
                try:
                    texture_img = Image.open(texture_path).convert("RGB").resize(pil_image.size)
                    pillow_mask = Image.fromarray(mask)
                    pil_image = Image.composite(texture_img, pil_image, pillow_mask)
                    
                    new_filename = f"{base_name}_texture_{i}{extension}"
                    fname = os.path.join("images", "suggestions", new_filename)
                    pil_image.save(fname)
                    suggestions_to_create.append(Suggestions(sugImg_id=img, sugImage=fname))
                except Exception as e:
                    print(f"Could not process texture {texture_name}: {e}")
        
        if suggestions_to_create:
            Suggestions.objects.bulk_create(suggestions_to_create)

    paths = Suggestions.objects.filter(sugImg_id=pk)
    return render(request, 'main/suggestions.html', {'pk': pk, 'img': img, 'paths': paths})