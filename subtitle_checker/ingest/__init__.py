"""Video ingest — audio track extraction and frame access.

Input: video file.
Output: audio track (wav) plus frame access for the subtitle stage.

Shared by every downstream stage; wraps ffmpeg/OpenCV so nothing else in the
package touches video decoding directly.
"""
