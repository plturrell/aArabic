from slide_generator import SlideGenerator, SlideRequest, PresentationConfig, SlideTheme
from collections import List

fn main():
    print("Testing Mojo Slide Generator...")
    
    # Initialize generator
    var generator = SlideGenerator("llama-3.2-1b", 0.7)
    
    # Prepare test data
    var source_ids = List[String]()
    source_ids.append(String("source_001"))
    source_ids.append(String("source_002"))
    
    var chunks = List[String]()
    chunks.append(String("Machine learning enables automated pattern recognition."))
    chunks.append(String("Deep learning uses multi-layer neural networks."))
    
    # Create request
    var config = PresentationConfig(
        SlideTheme.professional(),
        8,
        True,
        True,
        "executive",
        "medium"
    )
    
    var request = SlideRequest(
        source_ids,
        "Test Presentation",
        "Test Author",
        config,
        List[String]()
    )
    
    # Generate presentation
    var response = generator.generate_presentation(request, chunks)
    
    print("✓ Presentation generated successfully")
    print("  Slides: " + String(len(response.slides)))
    print("  Theme: " + response.theme.to_string())
