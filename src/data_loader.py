"""
Data loading and preprocessing utilities for Visual Question Answering (VQA) pipeline.
"""

from typing import Dict, Any, List, Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt


class VQADataset(Dataset):
    """
    Custom Dataset class for VQA tasks.
    
    Attributes:
        data (Union[List[Dict[str, Any]], Any]): List or HF Dataset of samples, each containing image, question, and answer.
        processor: Optional image transformations or Hugging Face processor.
    """
    def __init__(self, data: Union[List[Dict[str, Any]], Any], processor: Optional[Any] = None) -> None:
        self.data = data
        self.processor = processor

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image = item["image"]
        question = item["question"]
        
        # Extract answer
        if "multiple_choice_answer" in item and item["multiple_choice_answer"]:
            answer = item["multiple_choice_answer"]
        elif "answers" in item and item["answers"]:
            first_ans = item["answers"][0]
            answer = first_ans.get("answer", "") if isinstance(first_ans, dict) else str(first_ans)
        else:
            answer = item.get("answer", "")

        # If PIL image path is provided instead of PIL Image object
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if self.processor:
            inputs = self.processor(images=image, text=question, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["answer"] = answer
            return inputs

        return {
            "image": image,
            "question": question,
            "answer": answer
        }


def get_vqa_dataloader(
    data: Union[List[Dict[str, Any]], Any],
    processor: Optional[Any] = None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Creates and returns a PyTorch DataLoader for VQA dataset.
    """
    dataset = VQADataset(data, processor=processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_vqa_subset(split: str = "validation[:50]"):
    """
    Loads a light subset of the VQAv2 dataset from Hugging Face datasets.

    Args:
        split (str): Dataset split specification. Defaults to 'validation[:50]'.

    Returns:
        Dataset: Hugging Face Dataset split.
    """
    return load_dataset("HuggingFaceM4/VQAv2", split=split)


def run_eda(dataset: Any, num_samples: int = 3) -> None:
    """
    Performs Exploratory Data Analysis by plotting sample images with questions and top human answers.

    Args:
        dataset: Hugging Face Dataset or iterable VQA dataset.
        num_samples (int): Number of sample images to display. Defaults to 3.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    if num_samples == 1:
        axes = [axes]

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample["image"]
        question = sample.get("question", "No question")

        # Extract top human answer
        if "multiple_choice_answer" in sample and sample["multiple_choice_answer"]:
            answer = sample["multiple_choice_answer"]
        elif "answers" in sample and sample["answers"]:
            first_ans = sample["answers"][0]
            answer = first_ans.get("answer", "N/A") if isinstance(first_ans, dict) else str(first_ans)
        else:
            answer = sample.get("answer", "N/A")

        axes[i].imshow(image)
        axes[i].set_title(f"Q: {question}\nA: {answer}", fontsize=10, pad=10)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

