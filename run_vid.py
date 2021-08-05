'''
Takes telemetry data from RunKeeper and generates a 1080p compatible map video file to be an overlay on the bottom right.

1st version - S. Silburn - Generates video with overlay.

25.07.21 - P. Carvalho - Added command line arguments,
                        Uses ffprobe to determine the original video's parameters (make sure you have ffmpeg installed)
04.08.21                Uses ffmpeg to read in the original video
                        Enables encoding the full video with overlay included

e.g.: python run_vid.py -gpxpath D:\Docs\Dropbox\Personal\goPro\2021-06-05-121845.gpx -videopath "D:\Multimedia\IMGs\0.Fotos\2021\timeWarp_cycling\Bike Iffleyheadington-1.m4v" -timestretch 5 -delay 1 -outpath D:\Multimedia\IMGs\0.Fotos\2021\timeWarp_cycling\overlay.mp4


python run_vid.py -gpxpath D:\Docs\Dropbox\Personal\goPro\2021-06-05-121845.gpx -videopath "D:\Multimedia\IMGs\0.Fotos\2021\timeWarp_cycling\Bike Iffleyheadington-1-1.m4v" -timestretch 5 -delay 1 -wbitrate 100000 -outpath "D:\Multimedia\IMGs\0.Fotos\2021\timeWarp_cycling\end.mp4"


#use gpu:
python run_vid.py -gpxpath D:\Docs\Dropbox\Personal\goPro\2021-06-05-121845.gpx -videopath "D:\Multimedia\IMGs\0.Fotos\2021\timeWarp_cycling\Bike Iffleyheadington-1-1.m4v" -timestretch 5 -delay -2 -wbitrate 100000 -useGPU -outpath "D:\Multimedia\IMGs\0.Fotos\2021\timeWarp_cycling\end.mp4"

'''



import runkeeper
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import argparse
import subprocess
#from PIL import Image
import progressbar
import ffmpeg
import io
import GPUtil
import time


parser = argparse.ArgumentParser(description='Telemetry to Video Overlay.')
parser.add_argument('-gpxpath', type=str, help='.gpx File path', default=".\\RK_gpx _2019-02-23_1504.gpx")
parser.add_argument('-videopath', type=str, help='.mkv File path that has the video without overlay. Used to count the number of frames', default="D:\\Multimedia\\IMGs\\0.Fotos\\2021\\timeWarp_cycling\\Bike Iffleyheadington-1.m4v")
parser.add_argument('-delay', type=float, help='Delay between video and runKeeper time', default=0)
parser.add_argument('-timestretch', type=float, help='Time stretching factor for time lapse videos', default=1.)
parser.add_argument('-outpath', type=str, help='Output video file path', default=".\\test_video.mp4")
parser.add_argument('-wbitrate', type=int, help='Bitrate for the Output Video', default=15000)
parser.add_argument('-onlyOverlay', action="store_true", help='Set to only get the overlay on the output')
#parser.add_argument('-alpha', type=float, help='Alpha value used for blending the overlay (between 0 and 1). Defaults to 0.9.', default=0.9)
parser.add_argument('-showCoords', help='If set, displays the coordinates in the overlay.', action='store_true')
parser.add_argument('-useGPU', help='If set, uses GPU encoding.', action='store_true')

args = parser.parse_args()


#print("Supplied arguments: {}".format(args))



#run = runkeeper.Activity("E:\\Python\\RK_gpx _2019-02-23_1504.gpx")
run = runkeeper.Activity(args.gpxpath)

line = []
point = []


mapbg = run.map_bg
init_pos = run.get_position(0,units='map_pixels')


videoData = subprocess.Popen(['ffprobe', "-show_streams", "{}".format(args.videopath)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
output = str(videoData.stdout.read(), 'utf-8')
#wait for process to finish
stdout, stderr = videoData.communicate()
i=0
#print("START DATA")
output = ''.join(output)
output = output.split('\r\n')

#print(output[0])
video_params = {}
for line in output:
    param = line.split('=')
    #print("param> {}".format(param))
    if len(param) == 2:
        video_params[param[0]] = param[1]
        #print("{}. {} = {}".format(i, param[0], param[1]))
    #else:
    #    print("{}. {}".format(i, line))
    i+=1
#print("END DATA")
#Assuming only one video stream in the file with the required parameters
video_width = int(video_params['width'])
video_height = int(video_params['height'])
video_fps = video_params['avg_frame_rate']
if(len(video_fps.split('/')) == 2):
    fps_split = video_fps.split('/')
    videofps = int(fps_split[0]) / int(fps_split[1])
else:
    videofps = int(video_fps)
dt = 1./videofps
#print("videofps: {}",format(videofps))
totalframecount = int(video_params['nb_frames'])

my_dpi = 200

#print("mapbg size: {}".format(mapbg.size))
overlay_width = 0.25 if mapbg.size[0] >= mapbg.size[1] else 0.25*mapbg.size[0]/mapbg.size[1]
overlay_height = 0.25 if mapbg.size[0] < mapbg.size[1] else 0.25*mapbg.size[1]/mapbg.size[0]

# Output in 1080p
fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
#ax = fig.add_subplot(111)
#ax = plt.axes([0.95-overlay_width, 0.95-overlay_height, overlay_width, overlay_height])
#ax = fig.add_axes([0.95-overlay_width, 0.95-overlay_height, overlay_width, overlay_height], zorder=2)
ax = fig.add_axes([0.95-overlay_width, 0.02, overlay_width, overlay_height], zorder=2)
ax.imshow(mapbg)
plt.axis('off')


#ax.set_xlim([190,950])
#ax.set_ylim([320,90])
#plt.subplots_adjust(top=0.19,left=0.71,bottom=0.00,right=0.99)
#plt.subplots_adjust(top=0.3,left=0.7)
line = ax.plot(init_pos[0],init_pos[1],'r-',linewidth=3,alpha=0.5)[0]
point = ax.plot(init_pos[0],init_pos[1],'ro',markersize=5)[0]
if (args.showCoords):
    ax.set_title('Distance: 0.0 km \n pos xxx.yyy',fontweight='bold',color=(1,1,1))
else:
    ax.set_title('Distance: 0.0 km',fontweight='bold',color=(1,1,1))
#ax.set_xlabel('pos: 0,0', color=(1,1,1))
x = []
y = []

#plt.show()
#exit()

if (not args.onlyOverlay):
    #Open the original video file with ffmpeg
    #https://stackoverflow.com/questions/59998641/decode-and-show-h-264-chucked-video-sequence-with-python-from-pi-camera
    # Seek to stream beginning
    stream = io.BytesIO()
    stream.seek(0)


    # Use ffprobe to get video frames resolution
    ###############################################
    #p = ffmpeg.probe(args.videopath, select_streams='v');
    #p_width = p['streams'][0]['width']
    #p_height = p['streams'][0]['height']
    #p_n_frames = int(p['streams'][0]['nb_frames'])
    ###############################################


    # Stream the entire video as one large array of bytes
    ###############################################
    # https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md
    in_process = (
        ffmpeg
        .input(args.videopath)
        .video # Video only (no audio).
        .output('pipe:', format='rawvideo', pix_fmt='rgb24') #format='h264')#, crf=23)
        .run_async(pipe_stdout=True) # Run asynchronous, and stream to stdout
    )
    ###############################################

    # Open In-memory binary streams
    #stream = io.BytesIO(in_bytes)

    # Execute FFmpeg in a subprocess with sdtin as input pipe and stdout as output pipe
    # The input is going to be the video stream (memory buffer)
    # The ouptut format is raw video frames in BGR pixel format.
    # https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md
    # https://github.com/kkroening/ffmpeg-python/issues/156
    # http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
    #process = (
    #    ffmpeg
    #    .input('pipe:', format='rawvideo', pix_fmt='rgb24')
    #    .video
    #    .output('pipe:', format='rawvideo', pix_fmt='rgb24')    #, pix_fmt='bgr24')
    #    .run_async(pipe_stdin=True, pipe_stdout=True)
    #)

    # process.stdin.write(stream.getvalue())  # Write stream content to the pipe
    # process.stdin.close()  # close stdin (flush and send EOF)
    #outs, errs = process.communicate(input=stream.getvalue())

    position = 0
    
    
    #Initialize the background frame
    axbg = fig.add_axes([0, 0, 1, 1], zorder=1)
    axbg.axis('off')
    plt.xticks([])
    plt.yticks([])


#Determine which codec to use
codec = 'h264'
if args.useGPU:
    gpus = GPUtil.getGPUs() 
    gpu_name = gpus[0].name
    if gpu_name.split(" ")[0] == 'NVIDIA':
        codec = 'h264_nvenc'
    #sadly, GPUtil only detects Nvidia gpus.
    else:
         
        #AMD Radeon GPUs are a bit trickier (this would also work for Nvidia but GPUtil is so convenient)
        wmic_data = subprocess.Popen(['wmic', "path win32_VideoController get name"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output = str(wmic_data.stdout.read(), 'utf-8')
        #wait for process to finish
        stdout, stderr = wmic_data.communicate()
        i=0
        output = ''.join(output)
        output = output.split('\r\n')
        #print(output[0])
        gpu = {}
        for line in output:
            param = line.split(' ')
            if param[0] == "Radeon":
                codec = 'h264_vaapi'
                break
        
#Start the video writer
#if (args.onlyOverlay):
vid_writer = animation.FFMpegWriter(fps=videofps, bitrate=args.wbitrate)#, extra_args="-h encoder={}".format(codec))
animation.codec = codec
vid_writer.setup(fig,args.outpath,dpi=fig.dpi)
##A different way, maybe better
#else:
#    out_process = (
#        ffmpeg
#        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(video_width, video_height))
#        .output(args.outpath, vcodec='h264', video_bitrate=args.wbitrate)
#        .overwrite_output()
#        .run_async(pipe_stdin=True)
#    )


#set up the progress bar
if (args.onlyOverlay):
    progress = progressbar.ProgressBar(maxval=totalframecount, widgets=[progressbar.Bar('=', '[', ']', ' '), ' ', progressbar.Percentage()])
    progress.start()
    
start_time = time.time()

#Fig to bytes buffer
#io_buf = io.BytesIO()

for frame in range(totalframecount):
    st_frame_time = time.time()
    if (not args.onlyOverlay):
        st_vidframe_time = time.time()
        # Read raw video frame from stdout as bytes array.
        ##in_bytes = process.stdout.read(width * height * 3)
        #in_bytes = outs[position: position + video_width * video_height * 3]
        in_bytes = in_process.stdout.read(video_width*video_height*3)
        position += video_width * video_height * 3
        
        if not in_bytes:
            break

        # transform the bytes read into a numpy array
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([video_height, video_width, 3])
        )
        
        #in_image = Image.fromarray(in_frame, 'RGB') 
        #ax.imshow(in_frame)
        axbg.cla()
        axbg.axis('off')
        #plt.xticks([])
        #plt.yticks([])
        axbg.imshow(in_frame)

        end_vidframe_time = time.time()
        #print(">>>> getting the video frame took {:.2f} \n".format(end_vidframe_time - st_vidframe_time))
        
    # Conversion from frame number to real time of the runkeepter trace
    realtime = dt*args.timestretch*frame - args.delay   #0.736*frame - 1.7183

    if realtime >= 0 and realtime <= run.total_time:
        st_dist_time = time.time()
        #clear ax
        #ax.cla()
        
        dist = float(run.distance(realtime))
        x_,y_ = run.get_position(realtime,units='map_pixels')
        x.append(x_)
        y.append(y_)
        point.set_data(x_,y_)
        line.set_data(x,y)
        
        if (args.showCoords):
            lat, long = run.get_position(realtime,units='latlong')
            ax.set_title('Distance: {:.2f} km \n pos {:.5f},{:.5f} '.format(dist,lat, long),fontweight='bold',color=(1,1,1))
        else:
            ax.set_title('Distance: {:.2f} km'.format(dist),fontweight='bold',color=(1,1,1))
        
        end_dist_time = time.time()
        #print(">>>> map overlay update took {:.2f} \n".format(end_dist_time - st_dist_time))
    
    
    
    #plt.show()
    
    if (args.onlyOverlay):
        vid_writer.grab_frame(facecolor=(0.01,0.01,0.01))
        progress.update(frame+1)
    else:
        st_writeframe_time = time.time()
        #Convert figure to numpy array
        ##io_buf = io.BytesIO()
        #io_buf.seek(0)
        #fig.savefig(io_buf, format='raw', dpi=my_dpi)
        #end_savefig_time = time.time()
        #print(">>>>> io (@{}) savefig took {:.2f} \n".format(io_buf.tell(), end_savefig_time - st_writeframe_time))
        #io_buf.seek(0)
        #out_frame = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        #                 newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        ##io_buf.close()
        #end_buffer_time = time.time()
        #print(">>>> buffer reshape took {:.2f} \n".format(end_buffer_time - end_savefig_time))
        #print(">>>> fig to buffer took {:.2f} \n".format(end_buffer_time - st_writeframe_time))
        ##fig.canvas.draw()
        #print ("in_image size: {}".format(in_image.size))
        #im_overlay = Image.frombytes('RGBA', fig.canvas.get_width_height(),bytes(fig.canvas.buffer_rgba()))
        #print ("overlay image size: {}".format(im_overlay.size))        
        
        ##out_image = Image.blend(in_image, im_overlay, args.alpha)
        ##out_image = in_image.paste(im_overlay)
        
        
        vid_writer.grab_frame()
        
        #out_process.stdin.write(
        #    out_frame
        #    .astype(np.uint8)
        #    .tobytes()
        #)
        
        end_writeframe_time = time.time()
        #print(">>>> writing the frame took {:.2f} \n".format(end_writeframe_time - st_writeframe_time))
        
    #print("rendering {:6.3f}%".format(frame/totalframecount*100))
    #progress.update(frame+1)
    
    end_frame_time = time.time()
    #print(">>>> Rendering frame {} took {:.2f}. Total time elapsed {:.2f}".format(frame, end_frame_time-st_frame_time, end_frame_time - start_time))
    
    
#io_buf.close()


if (args.onlyOverlay):
    progress.finish()
    vid_writer.finish()
else:
    vid_writer.finish()
    #out_process.stdin.close()
    in_process.wait()
    #out_process.wait()



end_time = time.time()
print("Full processing took {:.2f}".format(end_time - start_time))
