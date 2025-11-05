import cv2
import numpy as np
import os

def show_image_fitted(window_name, img, max_width=900, max_height=700):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    disp = cv2.resize(img, (int(w*scale), int(h*scale)))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===============================================================
# 1️⃣ FUNCTION TO DETECT IF IMAGE IS GRAYSCALE OR COLOR
# ===============================================================
def is_grayscale(img, threshold=8):
    b, g, r = cv2.split(img)
    diff1 = cv2.absdiff(b, g)
    diff2 = cv2.absdiff(g, r)
    mean_diff = (np.mean(diff1) + np.mean(diff2)) / 2
    return mean_diff < threshold


# ===============================================================
# 2️⃣ GRAYSCALE DEFECT DETECTION (YOUR WORKING CODE)
# ===============================================================
def detect_bw_defects(img,
                      blur_ksize=(7,7),
                      adaptive_blocksize=51,
                      adaptive_C=7,
                      min_area_ratio=0.0002,
                      solidity_threshold=0.1,
                      defect_ratio_threshold=0.0001):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, blur_ksize, 0)

    mask = cv2.adaptiveThreshold(gray_blur, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,
                                 adaptive_blocksize, adaptive_C)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    min_area = max(1, int(min_area_ratio * img_area))

    good_contours = []
    total_defect_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 1
        solidity = float(area) / hull_area
        if solidity < solidity_threshold:
            continue
        good_contours.append(c)
        total_defect_area += area

    defect_ratio = total_defect_area / img_area
    label = "Defective" if defect_ratio > defect_ratio_threshold else "Non-Defective"
    return mask, good_contours, label, defect_ratio


# ===============================================================
# 3️⃣ COLOR DEFECT DETECTION (NEW LOGIC)
# ===============================================================
def detect_color_defects(img,
                         min_area_ratio=0.001,
                         solidity_threshold=0.3,
                         defect_ratio_threshold=0.0003):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # detect dark spots (holes) + faded/discolored regions
    dark_mask = cv2.inRange(v, 0, 70)
    bright_mask = cv2.inRange(v, 200, 255)
    low_sat_mask = cv2.inRange(s, 0, 40)

    # combine for all possible defect types
    mask = cv2.bitwise_or(dark_mask, low_sat_mask)
    mask = cv2.bitwise_or(mask, bright_mask)

    # emphasize hole edges
    edges = cv2.Laplacian(v, cv2.CV_64F)
    edges = cv2.convertScaleAbs(edges)
    _, edge_mask = cv2.threshold(edges, 35, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(mask, edge_mask)

    # morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    min_area = max(1, int(min_area_ratio * img_area))

    good_contours = []
    total_defect_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        if solidity < solidity_threshold:
            continue
        good_contours.append(c)
        total_defect_area += area

    defect_ratio = total_defect_area / img_area
    label = "Defective" if defect_ratio > defect_ratio_threshold else "Non-Defective"
    return mask, good_contours, label, defect_ratio


# ===============================================================
# 4️⃣ MAIN FUNCTION: AUTO SWITCH BASED ON IMAGE TYPE
# ===============================================================
def detect_fabric_defects(image_path,
                          resize_width=800,
                          save_result=True,
                          out_dir="output_images"):
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(image_path)

    h, w = orig.shape[:2]
    if w > resize_width:
        scale = resize_width / float(w)
        img = cv2.resize(orig, (resize_width, int(h * scale)))
    else:
        img = orig.copy()

    if is_grayscale(img):
        print("Detected as: Grayscale Image → using B&W pipeline")
        mask, contours, label, ratio = detect_bw_defects(img)
    else:
        print("Detected as: Color Image → using Color pipeline")
        mask, contours, label, ratio = detect_color_defects(img)

    vis = img.copy()
    cv2.drawContours(vis, contours, -1, (0,0,255), 2)
    font_scale = max(0.6, min(2.0, img.shape[1] / 800))
    text_color = (0, 0, 200) if label == "Defective" else (0, 200, 0)
    cv2.putText(vis, f"Classified: {label}", (10, int(30 * font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2 * font_scale, text_color, 2)

    if save_result:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(out_dir, f"{base}_result.png"), vis)
        cv2.imwrite(os.path.join(out_dir, f"{base}_mask.png"), mask)

    return vis, mask, label, ratio, contours


# ===============================================================
# 5️⃣ EXAMPLE USAGE
# ===============================================================
if __name__ == "__main__":
    path = r"C:\Users\chand\Downloads\11.jpg"  # change path

    vis, mask, label, ratio, contours = detect_fabric_defects(path)
    print("Label:", label, "| Defect Ratio:", ratio)
    show_image_fitted("Output", vis)
    show_image_fitted("Mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))