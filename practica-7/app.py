import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "google/mobilenet_v2_1.0_224"

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()


def clasificar(imagen):
    if imagen is None:
        return {}

    inputs = processor(images=imagen, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits[0], dim=-1)

    top3_probs, top3_indices = torch.topk(probs, k=3)

    resultados = {}
    for prob, idx in zip(top3_probs, top3_indices):
        etiqueta = model.config.id2label[idx.item()]
        resultados[etiqueta] = float(prob.item())

    return resultados


interfaz = gr.Interface(
    fn=clasificar,
    inputs=gr.Image(type="pil", label="Imagen de entrada"),
    outputs=gr.Label(num_top_classes=3, label="Top 3 predicciones"),
    title="Clasificador de Imágenes con MobileNetV2",
    description=f"Clasificación de imágenes utilizando el modelo `{MODEL_ID}`.",
)

if __name__ == "__main__":
    interfaz.launch()