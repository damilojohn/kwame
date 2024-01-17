# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

from transformers import GPT2LMHeadModel, GPT2Tokenizer


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.model =GPT2LMHeadModel.from_pretrained('gpt2') 
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def predict(
        self,
        text: str = Input(description="Enter any text here, try to be responsible"),
    ) -> str:
        """Run a single prediction on the model"""
        input = self.tokenizer(input,return_tensors='pt')
        output = self.model.generate(**input,)
        return self.tokenizer.decode(output[0])
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
