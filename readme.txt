################################

Takes telemetry data from a gpx file and generates a 1080p compatible map video file to be an overlay on the bottom right.

################################
Requirements:
Python
ffmpeg

######
python modules:
 - matplotlib
 - numpy
 - progressbar             2.5
 - ffmpeg-python           0.2.0
 - GPUtil                  1.4.0


################################

Usage:

> python -gpxpath {path} 				#.gpx File path
		-videopath {path} 				# .mkv File path that has the video without overlay
		[-delay {delay}] 				# Delay between video and runKeeper time
		[-timestretch {multiplier}] 	# Time stretching factor for time lapse videos
		[-outpath {path}] 				# Output video file path
		[-wbitrate {bps}] 				# Bitrate for the Output Video
		[-onlyOverlay]					# Set to only get the overlay on the output
		[-alpha {alpha}]				# Alpha value used for blending the overlay (between 0 and 1) {unusued}
		[-showCoords]					# If set, displays the coordinates in the overlay
		[-useGPU]						# If set, uses GPU for video encoding (not very useful, since most of the 
										##time is lost converting the matplotlib figure to numpy array)


################################

Updates:
1st version - S. Silburn - Generates video with overlay.
25.07.21 - P. Carvalho - Added command line arguments,
                        Uses ffprobe to determine the original video's parameters (make sure you have ffmpeg installed)
04.08.21                Uses ffmpeg to read in the original video
                        Enables encoding the full video with overlay included