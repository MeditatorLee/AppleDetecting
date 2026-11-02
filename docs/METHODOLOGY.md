# Methodology and modeling decisions

## Detection and localization

YOLOv5s predicts apples. Image-space box centers are converted to a bottom-left origin: `((xmin+xmax)/2, image_height-(ymin+ymax)/2)`. This reconstruction uses pretrained YOLOv5s fine-tuned for 60 epochs against historical pseudo-labels. The original competition paper instead used YOLOv5x.

## Maturity

Training images only: extract HSV values from an elliptical region within each pseudo-labeled bounding box. Construct 16-bin H and 16-bin S normalized histograms. Fit 3-way spectral clustering (fixed seed). For out-of-sample detections, use nearest saved cluster centroid; this is **not** native spectral-clustering prediction. User mapping: `0=Raw`, `1=Fully Mature`, `2=Half Mature`. H is circular; the offline report includes a color-coded polar histogram and separate color-filled S and V histograms, computed from training crops.

## Depth

Use official pretrained ZoeDepth-NK. For each detection, take the median finite positive depth over a central ellipse. A heatmap is normalized separately for each image, so colors are not comparable between images.

## Mass proxy

Let `SI=height/width`. If `0.5 <= SI <= 1.5`, use `R=(height+width)/2`; otherwise `R=max(height,width)`. Then `Rr=R/depth*scale`, `V=0.0002*Rr^3+0.0249*Rr^2`, `M_proxy=density*V`. Assumed densities: Raw=0.80, Fully Mature=0.88, Half Mature=0.84. Camera intrinsics, scene scale, and physical reference measurements are unknown, so **these are model-dependent proxies, not grams**.

The original paper is archived under `docs/original_report.pdf` for historical comparison. No fruit-recognition Question 5 workflow is included.
