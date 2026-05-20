# Data Labelling Process

Don't use labelImg, use x-anylabeling (https://github.com/CVHub520/X-AnyLabeling)\
1. Follow the installation instructions in the x-anylabeling repo
2. Launch x-anylabeling by running 'xanylabeling' in the terminal
- Make sure auto-save is on. Check 'File' in ribbon menu, there should be a green checkmark beside 'Save Automatically'
3. Click orange file icon at top left, open unlabelled image folder
4. Click orange AI button at bottom left, click 'No Model', then 'Load Custom Model'
5. Look for barrel_model.yaml, make sure best.onnx is in the same folder. Click on barrel_model.yaml
6. Click on orange play button above orange AI button. This will auto label all images in the folder.
7. Navigate between images using A (back) and D (next).
- If no annotation at all, press 'I' on keyboard, calls model to make inference
- If still missing annotation, press 'R' on keyboard, and draw annotation
- If wrong annotation, delete from 'Shape' list on right sidebar
8. When finished, press "Export" (on the ribbon menu), and export as VOC Detection for .xml files.


