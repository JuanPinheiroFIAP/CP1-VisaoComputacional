import cv2
import os
import matplotlib.pyplot as plt

path_image = os.path.join("image", "figura13.png")
image = cv2.imread(path_image,0)

# ============================
# Converter para cinza
# ============================
# def converter_para_cinza(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image_gray = converter_para_cinza(image)


# ============================
# Segmentação Otsu
# ============================
def apply_otsu(image):
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary

binary_otsu = apply_otsu(image)


# ============================
# Morfologia
# ============================
def morfologia(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return cleaned

cleaned = morfologia(binary_otsu)


# ============================
# Detectar contornos
# ============================
def detectar_contornos(imagem_binaria):
    contornos, _ = cv2.findContours(
        imagem_binaria,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Quantidade de contornos encontrados: {len(contornos)}")

    return contornos


# ============================
# Desenhar contornos
# ============================
def desenhar_contornos(imagem, contornos):

    imagem_copia = imagem.copy()

    cv2.drawContours(imagem_copia, contornos, -1, (0,255,0), 2)

    return imagem_copia


# ============================
# Bounding Boxes
# ============================
def desenhar_bounding_boxes(imagem, contornos):

    imagem_copia = imagem.copy()
    total_objetos = 0

    for contorno in contornos:

        # filtrar ruídos pequenos
        if cv2.contourArea(contorno) < 100:
            continue

        x, y, w, h = cv2.boundingRect(contorno)

        cv2.rectangle(
            imagem_copia,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        total_objetos += 1

    print("Quantidade total de objetos detectados:", total_objetos)

    return imagem_copia, total_objetos


# ============================
# Pipeline de processamento
# ============================

contornos_detectados = detectar_contornos(cleaned)

imagem_contornos = desenhar_contornos(image, contornos_detectados)

imagem_bbox, total_objetos = desenhar_bounding_boxes(image, contornos_detectados)


# ============================
# Plot do pipeline
# ============================
def plot_pipeline(image, binary_otsu, cleaned, imagem_bbox):
    bbox_rgb = cv2.cvtColor(imagem_bbox, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 5, figsize=(18,5))

    axes[0].imshow(image)
    axes[0].set_title("Imagem Original")
    axes[0].axis("off")

    axes[1].hist(image.ravel(), bins=156, range=[0,256])
    axes[1].set_title("Histograma")

    axes[2].imshow(binary_otsu, cmap="gray")
    axes[2].set_title("Segmentação Otsu")
    axes[2].axis("off")

    axes[3].imshow(cleaned, cmap="gray")
    axes[3].set_title("Morfologia")
    axes[3].axis("off")

    axes[4].imshow(bbox_rgb)
    axes[4].set_title(f"Bounding Boxes\nObjetos: {total_objetos}")
    axes[4].axis("off")

    fig.tight_layout()
    plt.show()


plot_pipeline(image, binary_otsu, cleaned, imagem_bbox)