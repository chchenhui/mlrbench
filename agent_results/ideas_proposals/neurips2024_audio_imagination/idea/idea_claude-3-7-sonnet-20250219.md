# Audio-to-Audio Style Transfer: Generating Music Across Genres

## Motivation
Music creation often involves drawing inspiration from existing pieces while adding personal creativity. However, transforming a composition from one genre to another (e.g., classical to jazz) requires significant musical expertise. An effective audio-to-audio style transfer system would democratize music creation, allowing musicians to easily experiment across genres while preserving core musical elements. This technology would also assist content creators in generating royalty-free music that maintains specific thematic elements while avoiding copyright issues through stylistic transformation.

## Main Idea
We propose a novel approach to music style transfer using a dual-encoder architecture with disentangled representation learning. The system first separates a source audio into content features (melody, harmony, rhythm) and style features (timbre, performance characteristics). A style encoder extracts characteristic features from target genre examples, which are then combined with the content features of the source audio through a decoder to generate new music. Unlike existing approaches that rely heavily on MIDI representations or symbolic music, our system works directly with raw audio, capturing nuanced performance elements. We employ contrastive learning to ensure style-content disentanglement and implement a multi-scale adversarial loss to maintain musical coherence at both local and global levels. The result is a flexible system that produces musically plausible transformations while preserving the essential musical identity of the original piece.