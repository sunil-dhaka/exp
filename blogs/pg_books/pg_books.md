---
layout: default
title: PG Books
has_children: true
---

# Automated Storybook Illustration: Leveraging Google's Imagen 3 for Consistent, High-Quality Book Art

## Project Overview

This project demonstrates an innovative approach to automated book illustration using Google's cutting-edge AI models. The system takes existing books from Project Gutenberg and transforms them into visually rich, illustrated storybooks by automatically generating character portraits and chapter illustrations that maintain consistent style and character appearance throughout the book.

The primary goal is to create a seamless pipeline that can:

1. Download and process public domain books from Project Gutenberg
2. Analyze the book content to identify key characters and scenes
3. Generate highly detailed prompts for image creation using Gemini models
4. Create consistent character illustrations across multiple scenes
5. Generate stylistically coherent chapter illustrations
6. Compile the illustrated book into a final product

## Tech Stack

The project is built on a focused, modern stack leveraging Google's latest AI models:

- **Google Imagen 3**: The backbone of our image generation pipeline, providing state-of-the-art visual rendering capabilities
- **Google Gemini models**: Used for natural language understanding and prompt engineering, including:
  - `gemini-2.0-pro-exp-02-05`: Primary model for sophisticated prompt generation
  - `gemini-2.0-flash`: Faster model for simpler text processing tasks
- **Python 3.10+**: Core programming language with modern type hints
- **Key Libraries**:
  - `google-genai`: Official Google Generative AI client library
  - `python-dotenv`: For secure environment variable management
  - `requests`: For API communication and book downloads
  - `PIL/Pillow`: For image processing

### Cost Considerations

An important aspect of this project was managing the cost of AI image generation. Using Imagen 3, the total cost for processing a complete book (~30 character illustrations and ~30 chapter illustrations) came to approximately **$50**. This cost breaks down to:

- Character illustrations: ~$0.85 per image
- Chapter illustrations: ~$0.80 per image
- Prompt generation via Gemini: Minimal cost impact

This represents a fraction of what professional book illustration would cost, while providing rapid results with consistent quality.

## Technical Architecture & Flow

The project follows a modular, pipeline-based architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│  Book Sourcing  │───▶│ Text Analysis & │───▶│ Image Generation│
│  & Processing   │    │ Prompt Creation │    │                 │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│  Final Book     │◀───│ Book Layout &   │◀───│ Image Post-     │
│  Compilation    │    │ PDF Generation  │    │ Processing      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Pipeline Components

#### 1. Book Sourcing & Text Processing

The first stage of the pipeline handles the acquisition and processing of literary content:

- **Content Acquisition Module**: Interfaces with Project Gutenberg's catalog to identify and download public domain books
- **Text Normalization Engine**: Cleans raw text by removing headers, footers, and formatting artifacts
- **Content Selection System**: Filters books based on language, length, and content suitability

#### 2. Content Analysis & Prompt Engineering

The second stage involves deep analysis of the book content to create effective prompts:

- **Character Identification System**: Analyzes the text to extract main characters and their descriptions
- **Scene Detection Module**: Identifies key scenes and plot points suitable for illustration
- **Prompt Generation Engine**: Uses Gemini AI to create detailed, context-rich prompts for image generation
  - Character prompt generator (50+ word detailed descriptions)
  - Scene composition prompt generator
  - Art style determination module

#### 3. Image Generation System

The third stage handles the actual creation of visual content:

- **Image Request Processor**: Configures and sends optimized requests to the Imagen API
- **Art Style Consistency Manager**: Ensures all generated images maintain the same artistic style
- **Character Recognition Validator**: Verifies character consistency across illustrations
- **Safety Filter System**: Applies content filtering to ensure appropriate imagery
- **Output Organization Module**: Systematically stores and categorizes generated images

#### 4. Image Post-Processing

The fourth stage refines the raw generated images:

- **Image Quality Enhancement**: Performs adjustments to brightness, contrast, and sharpness
- **Aspect Ratio Standardization**: Ensures consistent dimensions across illustrations
- **Metadata Embedding**: Attaches relevant information to images for downstream processing
- **Format Conversion**: Prepares images in appropriate formats for various output types

#### 5. Book Layout & Compilation

The fifth stage formats and arranges the content:

- **Text and Image Integration**: Combines the original text with appropriate illustrations
- **Layout Engine**: Creates a visually appealing arrangement of text and images
- **Format Production System**: Generates outputs in various formats (PDF, ePub, web)
- **Quality Assurance Module**: Performs final checks on the compiled product

#### 6. Media Expansion System

The final optional stage expands the content to other media formats:

- **Video Generation Module**: Creates animated content from illustrations
- **Audio-Visual Integration**: Combines narration with visuals for multimedia experiences
- **Distribution Format Converter**: Packages content for various platforms and devices

## The Magic of Effective Prompting

The exceptional consistency across illustrations is the result of sophisticated prompt engineering techniques. This is perhaps the most innovative aspect of the project. 

### Character Consistency Techniques

Our system achieves remarkable character consistency through:

1. **Detailed Character Description Caching**:
   - Initial detailed character descriptions (~50+ words) are generated
   - These descriptions include physical attributes, clothing, and distinctive features
   - Character descriptions are recycled and included in every prompt where that character appears

2. **Stylistic Consistency**:
   - A single art style is generated for the entire book and appended to every prompt
   - Style definitions include artistic technique, color palette, and mood indicators

3. **System Instructions**:
   - Every prompt includes consistent system instructions:
     - No text overlay in images
     - Family-friendly content guidelines
     - Avoidance of borders, titles, and extraneous elements

Here's an example of how a character prompt evolves:

**Initial Character Prompt:**
```
A tall man with sharp, angular features and deep-set blue eyes. He has silver-streaked dark hair swept back from a high forehead, and a neatly trimmed beard peppered with gray. His posture is commanding and upright, clothed in a tailored black suit with subtle pinstripes and a dark blue tie. His expression conveys wisdom and intensity, with slight crow's feet at the corners of his eyes suggesting both a capacity for warmth and years of serious contemplation.
```

**Chapter Illustration Prompt (including character):**
```
In a richly appointed Victorian library with floor-to-ceiling bookshelves and a crackling fireplace, Professor Harrington [Character Description from above] stands before an ancient map spread across a mahogany desk. Golden lamplight casts dramatic shadows as he points to a mysterious marking on the weathered parchment. Through the tall arched windows, the London fog presses against the glass while a brass telescope on a tripod gleams in the corner, suggesting the academic's interest in both terrestrial and celestial mysteries.
```

This systematic approach to prompt engineering ensures characters remain instantly recognizable across all illustrations while allowing for dynamic scene composition and emotional expression.

## Conclusion

This project demonstrates the remarkable potential of AI-powered illustration systems for book production. By leveraging the latest generation of image models and implementing sophisticated prompt engineering techniques, we've created a pipeline that produces consistent, high-quality illustrations at a fraction of the traditional cost.

While not yet replacing human illustrators, this system points toward fascinating possibilities for augmenting creative workflows, especially for backlist publishing, educational materials, and rapid prototyping of visual narratives.

---

*This project was created using Google Gemini and Imagen 3 APIs with a total illustration cost of approximately $50 per book.* 