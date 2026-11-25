from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


class HuggingFaceProvider:

    @staticmethod
    def create_llm():
        import torch

        device = 0 if torch.cuda.is_available() else -1
        model_kwargs = {"dtype": "auto"} if device >= 0 else {}
        pipeline = HuggingFacePipeline.from_model_id(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            task="text-generation",
            device=device,
            model_kwargs=model_kwargs,
            pipeline_kwargs={
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.2,
                "top_p": 0.9,
                "return_full_text": False,
            },
        )
        return ChatHuggingFace(llm=pipeline)
