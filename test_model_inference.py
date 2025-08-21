import logging
import sys
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List, Dict, Any

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append('.')
sys.path.append('..')

class ConversationalQueryRewriter:
    """
    Class để xử lý query rewriting từ hội thoại sử dụng model KD-ANCE-prefix-oracle-best-model
    """
    
    def __init__(self, model_path: str = "KD-ANCE-prefix-oracle-best-model", device: str = None):
        """
        Khởi tạo model và tokenizer
        
        Args:
            model_path: Đường dẫn đến model đã train
            device: Device để chạy model (cuda/cpu)
        """
        self.model_path = model_path
        
        # Tự động phát hiện device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Sử dụng device: {self.device}")
        
        # Load tokenizer và model
        self._load_model()
        
        # Thiết lập các tham số mặc định
        self.max_query_length = 32
        self.max_response_length = 64
        self.max_concat_length = 512
        self.use_prefix = True
        
    def _load_model(self):
        """Load model và tokenizer từ checkpoint"""
        try:
            logger.info(f"Đang load model từ {self.model_path}...")
            
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Load model thành công!")
            
        except Exception as e:
            logger.error(f"Lỗi khi load model: {e}")
            raise
    
    def _padding_seq_to_same_length(self, input_ids: List[int], max_pad_length: int, pad_token: int = 0):
        """
        Padding sequence đến độ dài cố định
        
        Args:
            input_ids: List token IDs
            max_pad_length: Độ dài tối đa sau padding
            pad_token: Token dùng để padding
            
        Returns:
            Tuple (padded_input_ids, attention_mask)
        """
        padding_length = max_pad_length - len(input_ids)
        attention_mask = []

        if padding_length <= 0:
            attention_mask = [1] * max_pad_length
            input_ids = input_ids[:max_pad_length]
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + [pad_token] * padding_length

        return input_ids, attention_mask
    
    def _prepare_input(self, conversation_history: List[str], current_query: str) -> Dict[str, torch.Tensor]:
        """
        Chuẩn bị input cho model từ lịch sử hội thoại và query hiện tại
        
        Args:
            conversation_history: List các câu trong lịch sử hội thoại
            current_query: Câu query hiện tại
            
        Returns:
            Dict chứa input_ids và attention_mask
        """
        flat_concat = []
        
        # Thêm prefix cho current query nếu cần
        if self.use_prefix:
            current_query_text = "question: " + current_query
        else:
            current_query_text = current_query
            
        # Encode current query
        cur_utt = self.tokenizer.encode(
            current_query_text, 
            add_special_tokens=True, 
            max_length=self.max_query_length
        )
        flat_concat.extend(cur_utt)
        
        # Xử lý lịch sử hội thoại (từ cuối lên đầu)
        first_context = True
        for j in range(len(conversation_history) - 1, -1, -1):
            # Xác định max_length dựa trên vị trí (query hoặc response)
            if j % 2 == 1:  # Response
                max_length = self.max_response_length
            else:  # Query
                max_length = self.max_query_length
                
            # Thêm prefix cho context đầu tiên
            context_text = conversation_history[j]
            if self.use_prefix and first_context:
                context_text = "context: " + context_text
                first_context = False
                
            # Encode utterance
            utt = self.tokenizer.encode(
                context_text,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True
            )
            
            # Kiểm tra độ dài tối đa
            if len(flat_concat) + len(utt) > self.max_concat_length:
                remaining_length = self.max_concat_length - len(flat_concat) - 1
                if remaining_length > 0:
                    flat_concat += utt[:remaining_length] + [utt[-1]]  # Phải kết thúc bằng [SEP]
                break
            else:
                flat_concat.extend(utt)
        
        # Padding
        flat_concat, flat_concat_mask = self._padding_seq_to_same_length(
            flat_concat, max_pad_length=self.max_concat_length
        )
        
        return {
            "input_ids": torch.tensor([flat_concat], dtype=torch.long).to(self.device),
            "attention_mask": torch.tensor([flat_concat_mask], dtype=torch.long).to(self.device)
        }
    
    def generate_summary_query(self, conversation_history: List[str], current_query: str) -> str:
        """
        Tạo summary query từ lịch sử hội thoại và query hiện tại
        
        Args:
            conversation_history: List các câu trong lịch sử hội thoại
                                 Thứ tự: [query1, response1, query2, response2, ...]
            current_query: Câu query hiện tại cần được rewrite
            
        Returns:
            Summary query đã được rewrite
        """
        try:
            # Chuẩn bị input
            inputs = self._prepare_input(conversation_history, current_query)
            
            # Generate
            with torch.no_grad():
                output_seqs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=False,
                    max_length=self.max_query_length,
                    num_return_sequences=1
                )
            
            # Decode output
            summary_query = self.tokenizer.decode(output_seqs[0], skip_special_tokens=True)
            
            logger.info(f"Query gốc: {current_query}")
            logger.info(f"Summary query: {summary_query}")
            
            return summary_query
            
        except Exception as e:
            logger.error(f"Lỗi khi generate summary query: {e}")
            return current_query  # Trả về query gốc nếu có lỗi

def test_model_inference():
    """
    Hàm test model với dữ liệu mẫu
    """
    # Khởi tạo model
    rewriter = ConversationalQueryRewriter()
    
    # Dữ liệu test mẫu
    test_cases = [
        {
            "conversation_history": [
                "What is the capital of France?",
                "The capital of France is Paris.",
                "How many people live there?"
            ],
            "current_query": "What about the weather?"
        },
        {
            "conversation_history": [
                "Tell me about machine learning",
                "Machine learning is a subset of artificial intelligence that involves training algorithms on data.",
                "What are some applications?"
            ],
            "current_query": "How does it work?"
        },
        {
            "conversation_history": [],
            "current_query": "What is the weather like today?"
        }
    ]
    
    print("=== Test Model Inference ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test case {i}:")
        print(f"Conversation history: {test_case['conversation_history']}")
        print(f"Current query: {test_case['current_query']}")
        
        summary_query = rewriter.generate_summary_query(
            test_case['conversation_history'], 
            test_case['current_query']
        )
        
        print(f"Summary query: {summary_query}")
        print("-" * 80)
        print()

if __name__ == "__main__":
    test_model_inference()