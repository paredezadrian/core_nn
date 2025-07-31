# Known Limitations and Design Considerations

This document outlines known limitations in the current CORE-NN implementation that are by design or require future improvements.

## Interactive Chat Input Handling

### Tokenization Limitations

The current CLI implementation uses a simplified tokenization approach that has several intentional limitations:

#### Current Implementation
- **Character-based tokenization**: Converts each character to its ASCII value (capped at 999)
- **Fixed input length**: Limited to 50 characters maximum
- **No vocabulary mapping**: No proper token vocabulary or subword tokenization
- **ASCII-only output**: Detokenization only handles printable ASCII characters (32-126)
- **Fixed padding**: Always pads to maximum length regardless of input

#### Impact
- Input text longer than 50 characters is truncated
- Non-ASCII characters may not be handled correctly
- Generated text may appear garbled due to character-level reconstruction
- No handling of special tokens, punctuation, or linguistic structures

#### Status
**This is a design limitation, not a bug.** The simplified tokenization is intentionally used for demonstration purposes and rapid prototyping.

#### Recommended Solutions for Production
1. **Hugging Face Transformers Tokenizer**
   ```python
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained("gpt2")
   ```

2. **SentencePiece Tokenizer**
   ```python
   import sentencepiece as spm
   sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
   ```

3. **Custom Vocabulary-based Tokenizer**
   - Build vocabulary from training data
   - Implement subword tokenization (BPE, WordPiece)
   - Handle out-of-vocabulary tokens properly

### Memory and Context Limitations

#### Session Memory
- Sessions are stored locally and not distributed
- No automatic session cleanup or archiving
- Memory usage grows with session length

#### Context Window
- **IMPROVED**: Extended maximum sequence length support (up to 8192 tokens)
- **IMPROVED**: Chunked processing for very long sequences
- **IMPROVED**: Memory-efficient processing with automatic cleanup
- **IMPROVED**: Adaptive memory management within 10GB limits

## Model Architecture Limitations

### Simplified Components
Several components use simplified implementations suitable for edge devices but may not match state-of-the-art performance:

1. **Attention Mechanisms**: Simplified attention in some components
2. **Position Encoding**: **IMPROVED** - Hybrid learned + sinusoidal encoding for long sequences
3. **Layer Normalization**: Standard LayerNorm without advanced variants

### Memory Constraints
- **IMPROVED**: Optimized for edge devices with 10GB memory limits
- **IMPROVED**: 77.9% parameter reduction while maintaining performance
- **IMPROVED**: Memory-efficient processing with automatic cleanup
- Trade-offs between efficiency and performance

## Configuration and Deployment

### Device Support
- Primary focus on CPU and basic GPU support
- Limited optimization for specialized hardware (TPUs, custom accelerators)
- No distributed inference support

### Model Formats
- Currently uses PyTorch native format
- No ONNX or TensorRT optimization
- Limited quantization support

## Future Improvements

### Planned Enhancements
1. **Advanced Tokenization**: Integration with production-ready tokenizers
2. **Context Management**: Sliding window and context compression
3. **Hardware Optimization**: Better GPU utilization and quantization
4. **Distributed Support**: Multi-device inference capabilities

### Community Contributions
We welcome contributions to address these limitations. Please see the contributing guidelines for more information.

## Workarounds

### For Tokenization Issues
1. Keep input messages under 50 characters when using CLI
2. Use the Python API directly with proper tokenizers
3. Implement custom tokenization wrapper

### For Memory Issues
1. Regularly save and clear sessions
2. Monitor memory usage in long conversations
3. Adjust configuration parameters for your hardware

### For Performance Issues
1. Tune configuration parameters for your use case
2. Use appropriate device settings (CPU vs GPU)
3. Enable mixed precision if supported

## Reporting Issues

If you encounter behavior that seems like a bug but might be a known limitation:

1. Check this document first
2. Review the configuration options
3. Check the code comments for "DESIGN LIMITATION" markers
4. Open an issue with detailed reproduction steps

Remember: Not all unexpected behavior is a bug - some may be intentional design decisions for edge device compatibility.
