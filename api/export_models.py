import os
import torch
import torch.nn as nn
from ultralytics import YOLO
from mobile_sam import sam_model_registry

def export_yolo(model_path: str = "yolo11n.pt", output_path: str = "yolo11n.onnx"):
    print(f"Exporting YOLO model: {model_path} -> {output_path}")
    model = YOLO(model_path)
    model.export(format="onnx", dynamic=True, simplify=True, opset=18)

class MobileSAMDecoderWrapper(nn.Module):
    """Wrapper to decouple the lightweight SAM decoder from PyTorch internals."""
    def __init__(self, sam_model):
        super().__init__()
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder

    def forward(self, image_embeddings, boxes):
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks

def export_mobile_sam(checkpoint_path: str = "mobile_sam.pt", model_type: str = "vit_t"):
    print("Exporting MobileSAM Encoder and Decoder to ONNX...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.eval()

    # 1. Export Image Encoder using standard PyTorch tracing
    encoder = sam.image_encoder
    dummy_input = torch.randn(1, 3, 1024, 1024)
    torch.onnx.export(
        encoder,
        dummy_input,
        "sam_encoder.onnx",
        input_names=["images"],
        output_names=["image_embeddings"],
        dynamic_axes={"images": {0: "batch_size"}},
        opset_version=18,
        dynamo=False,  # Enforce stable legacy tracing exporter
    )
    print("Exported: sam_encoder.onnx")

    # 2. Export Mask Decoder using standard PyTorch tracing
    decoder_wrapper = MobileSAMDecoderWrapper(sam)
    dummy_embedding = torch.randn(1, 256, 64, 64)
    dummy_boxes = torch.tensor([[[100.0, 100.0, 200.0, 200.0]]], dtype=torch.float32)

    torch.onnx.export(
        decoder_wrapper,
        (dummy_embedding, dummy_boxes),
        "sam_decoder.onnx",
        input_names=["image_embeddings", "boxes"],
        output_names=["masks"],
        dynamic_axes={
            "image_embeddings": {0: "batch_size"},
            "boxes": {0: "batch_size"},
            "masks": {0: "batch_size"},
        },
        opset_version=18,
        dynamo=False,  # Enforce stable legacy tracing exporter
    )
    print("Exported: sam_decoder.onnx")

if __name__ == "__main__":
    export_yolo()
    export_mobile_sam()
    print("\nAll models successfully exported to ONNX format without errors!")