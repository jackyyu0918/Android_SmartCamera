package com.jackyyu0918.finalyearproject_54820425;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.TypedValue;
import android.view.MenuItem;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.appcompat.widget.SwitchCompat;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.Tracker;
import org.opencv.tracking.TrackerBoosting;
import org.opencv.tracking.TrackerCSRT;
import org.opencv.tracking.TrackerGOTURN;
import org.opencv.tracking.TrackerKCF;
import org.opencv.tracking.TrackerMIL;
import org.opencv.tracking.TrackerMOSSE;
import org.opencv.tracking.TrackerMedianFlow;
import org.opencv.tracking.TrackerTLD;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;

//TensorflowLite library
import com.jackyyu0918.finalyearproject_54820425.customview.OverlayView;
import com.jackyyu0918.finalyearproject_54820425.tflite.*;
import com.jackyyu0918.finalyearproject_54820425.env.*;
import com.jackyyu0918.finalyearproject_54820425.tracking.MultiBoxTracker;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";

    //private CameraBridgeViewBase mOpenCvCameraView;
    private Zoomcameraview mOpenCvCameraView;
    private boolean              mIsJavaCamera = true;
    private MenuItem             mItemSwitchCamera = null;

    //--------------------Class Field-----------------//
    //View object
    private DragRegionView DragRegionView;
    OverlayView trackingOverlay;

    //Matrix
    private Mat mRgba;

    private Mat mGray;
    private Mat testMat;
    private Mat videoMat;

    private Mat targetObjectMat;
    private Mat zoomWindowMat;
    private Mat optimizeObjectMat;

    //Tracker
    private Tracker objectTracker;
    private boolean isInitTracker = false;
    //TensorFlow MultiBoxTracker
    private MultiBoxTracker multiBoxTracker;

    private Rect2d roiRect2d;
    private Rect roiRect;
    //Tracking object use rect
    private Rect targetObjectRect;

    //Pre-defined size
    private Point trackerCoordinate;
    private Point trackerSize;
    private Scalar greenColorOutline;


    //Mode switching
    //false = small window
    private boolean isFullScreen = false;

    //button
    // for init tracker
    private Button button_startTrack;
    private String currentTrackerType;

    // for reset tracker
    private Button button_resetTrack;

    // for full view tracking
    private Button button_fullView;

    // for start recording
    private Button button_startRecord;

    //Media recorder
    public MediaRecorder recorder = new MediaRecorder();
    private boolean isRecording = false;

    //Switch for mode
    //true = ManualMode (Drag action), false = Automatic mode (Object detection)
    private boolean ManualMode = false;

    //=================TensorFlowLite interpreter==================//
        //Camera Activity
    private int[] rgbBytes = null;
    private boolean isProcessingFrame = false;

    protected int previewWidth = 0;
    protected int previewHeight = 0;
    private boolean useCamera2API;
    private byte[][] yuvBytes = new byte[3][];
    private int yRowStride;

    //Thread handler
    private Handler handler;
    private HandlerThread handlerThread;

    //Work for thread, defined at the onPreviewFrame
    private Runnable postInferenceCallback;
    private Runnable imageConverter;

    private SwitchCompat apiSwitchCompat;

        //Detector Activity
    private static final Logger LOGGER = new Logger();

    // Configuration values for the prepackaged SSD model.
    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final boolean TF_OD_API_IS_QUANTIZED = true;
    //model name
    private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection. 0.5/50%
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    //private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
    private static final boolean MAINTAIN_ASPECT = false;

    //Hard coded size!? it affect the size of preview, which is the same output as my project
    //private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    //phone with 2:1 apsect ratio
    private static final android.util.Size DESIRED_PREVIEW_SIZE = new android.util.Size(1920, 960);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    //Sensor Orientation
    private Integer sensorOrientation;

    private Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private BorderedText borderedText;

    //
    private Camera camera;
    private byte[] bytes;

    private boolean debug = false;

    //private SwitchCompat apiSwitchCompat;


    //----------------------------end of class field----------------------------------//

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        //Hide bar
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);


        //default setting
        /*
        mOpenCvCameraView = findViewById(R.id.main_surfaceView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
         */

        //Zoom view setting
        mOpenCvCameraView = (Zoomcameraview)findViewById(R.id.ZoomCameraView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setZoomControl((SeekBar) findViewById(R.id.CameraZoomControls));
        mOpenCvCameraView.setCvCameraViewListener(this);

        //Overlay view setting


        //Grant permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 1888);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.WRITE_EXTERNAL_STORAGE}, 112);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.RECORD_AUDIO}, 120);
        }

        //Tracker section
        //roiRect2d setting

        //old square roiRect2d 1:1
        /*
        trackerCoordinate = new Point(700,200);
        trackerSize = new Point(300,300);
        greenColorOutline = new Scalar(0, 255, 0, 255);
         */


        //new 2:1 size
        /*
        trackerCoordinate = new Point(1000,200);
        trackerSize = new Point(210,300);
        greenColorOutline = new Scalar(0, 255, 0, 255);
        roiRect2d = new Rect2d(trackerCoordinate.x,trackerCoordinate.y,trackerSize.x,trackerSize.y);
        roiRect = new Rect((int)trackerCoordinate.x,(int)trackerCoordinate.y,(int)trackerSize.x,(int)trackerSize.y);
         */

        greenColorOutline = new Scalar(0, 255, 0, 255);
        trackerCoordinate = new Point();
        trackerSize = new Point();

        roiRect2d = new Rect2d();
        roiRect = new Rect();
        //tracking rect
        targetObjectRect = new Rect();

        //tracker creation, base on drop down menu selection
        //currentTrackerType = "KCF";

        //spinner tracker selection
        final Spinner trackerSpinner  = findViewById(R.id.trackerSpinner);

        final ArrayAdapter<String> nameAdaptar = new ArrayAdapter<String>(MainActivity.this,android.R.layout.simple_expandable_list_item_1, getResources().getStringArray(R.array.trackingAlgorithmName));

        nameAdaptar.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        trackerSpinner.setAdapter(nameAdaptar);


        trackerSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                Toast.makeText(MainActivity.this, "You are choosing "+ parent.getSelectedItem().toString() + ".", Toast.LENGTH_SHORT).show() ;
                currentTrackerType = parent.getSelectedItem().toString();
                System.out.println("currentTrackerType: " + currentTrackerType);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                Toast.makeText(MainActivity.this, "Nothing is selected.", Toast.LENGTH_LONG).show();
            }
        });

        //button onClick listener
        //start button
        button_startTrack = findViewById(R.id.button_startTrack);
        button_startTrack.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view)
            {
                //No default tracker type
                //firstTracker = TrackerKCF.create();

                //own function, create proper tracker
                if (DragRegionView.points[0] == null){
                    Toast toast1 = Toast.makeText(MainActivity.this,
                            "Please drag on target object.", Toast.LENGTH_LONG);
                    //顯示Toast
                    toast1.show();
                } else {
                    createTracker(currentTrackerType);

                    //get user drag result
                    calculateRectInfo(DragRegionView.points);

                    //tracker initialization
                    objectTracker.init(mGray, roiRect2d);
                    //System.out.println("Tracker init result: " + firstTracker.init(mGray,roiRect2d));
                    isInitTracker = true;

                    //Tracker message
                    Toast toast1 = Toast.makeText(MainActivity.this,
                            "Current tracker: " + objectTracker.getClass(), Toast.LENGTH_LONG);
                    //顯示Toast
                    toast1.show();

                    Toast toast2 = Toast.makeText(MainActivity.this,
                            "Current camera size: " + mOpenCvCameraView.getWidth() + "x" + mOpenCvCameraView.getHeight(), Toast.LENGTH_LONG);
                    //顯示Toast
                    toast2.show();

                    DragRegionView.isReset = true;
                    DragRegionView.invalidate();

                    Toast toast3 = Toast.makeText(MainActivity.this,
                            "Current tracker size: " + roiRect.width + "x" + roiRect.height, Toast.LENGTH_LONG);
                    //顯示Toast
                    toast3.show();
                }
            }
        });

        //switch mode button
        button_fullView = findViewById(R.id.button_fullView);
        button_fullView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view)
            {
                if(isFullScreen == false) {
                    isFullScreen = true;
                } else {
                    isFullScreen = false;
                }
                //Tracker message
                Toast toast = Toast.makeText(MainActivity.this,
                        "Full screen mode: " + isFullScreen , Toast.LENGTH_LONG);
                //顯示Toast
                toast.show();
            }
        });

        //reset button
        button_resetTrack = findViewById(R.id.button_resetTracker);
        button_resetTrack.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view)
            {
                isInitTracker = false;
                objectTracker = null;

                //new 2:1 size
                resetTracker();
                DragRegionView.isReset = true;
                DragRegionView.invalidate();
                //testing****
                //System.out.println("Coordinate: " + DragRegionView.points[0] + DragRegionView.points[1] + DragRegionView.points[2] + DragRegionView.points[3]);
            }
        });

        //start recording button
        //start button
        button_startRecord = findViewById(R.id.button_startRecord);
        button_startRecord.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view)
            {
                // prepareRecord();  <-- Moved to onCreate()

                if(isRecording == false){
                    recorder.reset();
                    prepareRecorder();

                    //Recording message
                    Toast toast = Toast.makeText(MainActivity.this,
                            "Start recording...", Toast.LENGTH_LONG);
                    //顯示Toast
                    toast.show();
                    button_startRecord.setText("STOP RECORDING");
                    startRecord();
                } else {
                    //Recording message
                    Toast toast = Toast.makeText(MainActivity.this,
                            "End recording...", Toast.LENGTH_LONG);
                    //顯示Toast
                    toast.show();
                    button_startRecord.setText("START RECORDING");

                    //empty the recorder before stop
                    mOpenCvCameraView.setRecorder(null);
                    stopRecord();
                }
            }
        });

        //Sensor View at top
        DragRegionView = (DragRegionView) findViewById(R.id.SensorView);
    }

    @Override
    public void onPause()
    {
        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            System.out.println(e);
        }

        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }



        //Multi Thread issue -Me
        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

        //Media recorder;
        //recorder.release();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
        testMat =  new Mat();
        targetObjectMat = new Mat();
        zoomWindowMat = new Mat();
        optimizeObjectMat = new Mat();

        //Camera parameter
        camera = mOpenCvCameraView.getmCamera();
        bytes = mOpenCvCameraView.getmBuffer();

        //Initialize tracker
        multiBoxTracker = new MultiBoxTracker(this);
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    //===============onCameraPreiew=================//

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        //Basic Info
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        Size sizeRgba = mRgba.size();
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;

        //===============tracking section==============//
        /*
        // if initialized tracker, start update the ROI
        if(isInitTracker){

            //Pre-defined target window details: x,y,width,height
            //Assign 2d to 1d:
            // 2d: update by tracker
            // 1d: update Rec
            roiRect.x = (int) roiRect2d.x;
            roiRect.y = (int) roiRect2d.y;
            roiRect.width = (int) roiRect2d.width;
            roiRect.height = (int) roiRect2d.height;

            System.out.println("x,y,width,height: " + (int) roiRect2d.x + ", "+ (int) roiRect2d.y + ", "+ (int) roiRect2d.width + ", "+ (int) roiRect2d.height);


            //Update tracker information to roiRect2d
            //Why not multi thread, high latency
            System.out.println("Tracker update result: " + objectTracker.update(mGray, roiRect2d));

            //make sure target object is inside the screen
            if(roiRect2d.x+ roiRect2d.width > 3000 || roiRect2d.x < 0 || roiRect2d.y < 0 || roiRect2d.y+ roiRect2d.height > 1080) {
                System.out.println("Tracking Failed, target object fall outside screen");

            } else {
                //Target object matrix frame
                targetObjectMat = mRgba.submat((int)(roiRect2d.y), (int)(roiRect2d.y+ roiRect2d.height), (int)(roiRect2d.x), (int)(roiRect2d.x+ roiRect2d.width));

                if(roiRect2d.height >= roiRect2d.width) {
                    //Optimized aspect ratio for video recording (2:1)
                    System.out.println("Before crash: " + (int) (roiRect2d.y) + ", " + (int) (roiRect2d.y + roiRect2d.height) + ", " + (int) (roiRect2d.x + (roiRect2d.width/2) - roiRect2d.height) + ", " + (int) (roiRect2d.x + (roiRect2d.width/2) + roiRect2d.height));
                    optimizeObjectMat = mRgba.submat((int) (roiRect2d.y), (int) (roiRect2d.y + roiRect2d.height), (int) (roiRect2d.x + (roiRect2d.width/2) - roiRect2d.height), (int) (roiRect2d.x + (roiRect2d.width/2) + roiRect2d.height));
                } else if(roiRect2d.height < roiRect2d.width){
                    //Optimized aspect ratio for video recording (2:1)
                    optimizeObjectMat = mRgba.submat((int)(roiRect2d.y + (roiRect2d.height/2) - roiRect2d.width/4), (int)((roiRect2d.y + (roiRect2d.height/2) - roiRect2d.width/4) + roiRect2d.width/2), (int)(roiRect2d.x), (int)(roiRect2d.x+ roiRect2d.width));
                }

                // Small window preview mode
                if(isFullScreen == false){
                    //top-left corner of mRgba
                    zoomWindowMat = mRgba.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);

                    //show target matrix at the top-left corner
                    //Imgproc.resize(targetObjectMat, zoomWindowMat, zoomWindowMat.size(), 0, 0, Imgproc.INTER_LINEAR_EXACT);
                    Imgproc.resize(optimizeObjectMat, zoomWindowMat, zoomWindowMat.size(), 0, 0, Imgproc.INTER_LINEAR_EXACT);

                    // Full screen mode (for video recording)
                } else {
                    mRgba.copyTo(testMat);

                    //full the screen with target matrix
                    //Imgproc.resize(targetObjectMat, testMat, mRgba.size(), 0, 0, Imgproc.INTER_LINEAR_EXACT);
                    Imgproc.resize(optimizeObjectMat, testMat, mRgba.size(), 0, 0, Imgproc.INTER_LINEAR_EXACT);
                    //draw on full screen
                    Imgproc.rectangle(testMat,roiRect,greenColorOutline,2);
                    return testMat;
                }
            }

            Imgproc.rectangle(mRgba,roiRect,greenColorOutline,2);
            Imgproc.putText(mRgba,"Target",new Point(roiRect.x,roiRect.y),1,3,greenColorOutline);
        }
        */

        //============End of normal tracking==============//

        /*
        //================Object detection================//
        //CameraActivity
        if (isProcessingFrame) {
            LOGGER.w("Dropping frame!");
            return mRgba;
        }

        try {
            // Initialize the storage bitmaps once when the resolution is known.
            //Set up rgbBytes with camera size
            //If already run setup, no need to run this setup
            if (rgbBytes == null) {
                Camera.Size previewSize = camera.getParameters().getPreviewSize();
                previewHeight = previewSize.height;
                previewWidth = previewSize.width;
                rgbBytes = new int[previewWidth * previewHeight];

                //onPreviewSizeChosen
                onPreviewSizeChosen(new android.util.Size(previewSize.width, previewSize.height), 90);
            }

        } catch (final Exception e) {
            LOGGER.e(e, "Exception!");
            return mRgba;
        }

        isProcessingFrame = true;
        yuvBytes[0] = bytes;
        yRowStride = previewWidth;//Debug


        // Convert byes to rbgBytes
        imageConverter =
                new Runnable() {
                    @Override
                    public void run() {
                        //Store
                        ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);
                    }
                };


        postInferenceCallback =
                new Runnable() {
                    @Override
                    public void run() {
                        camera.addCallbackBuffer(bytes);
                        isProcessingFrame = false;
                    }
                };

        //Process Image ()
        processImage();

        //Return colorful Matrix to generate camera preview

         */

        //Imgproc.rectangle(mRgba, new Rect(786, 245, 1303-786, 663-245),greenColorOutline,2);

        return mRgba;
    }

    //================End of onCameraPreview=========//

    //my own function
    public void createTracker(String trackerType){
        switch (trackerType){
            case "KCF":
                System.out.println("KCF case.");
                objectTracker = TrackerKCF.create();
                break;
            case "MedianFlow":
                System.out.println("MedianFlow case.");
                objectTracker = TrackerMedianFlow.create();
                break;
            case "TLD":
                System.out.println("TLD case.");
                objectTracker = TrackerTLD.create();
                break;
            case "Boosting":
                System.out.println("Boosting case.");
                objectTracker = TrackerBoosting.create();
                break;
            case "CSRT":
                System.out.println("CSRT case.");
                objectTracker = TrackerCSRT.create();
                break;
            case "GOTURN":
                System.out.println("GOTURN case.");
                objectTracker = TrackerGOTURN.create();
                break;
            case "MIL":
                System.out.println("MIL case.");
                objectTracker = TrackerMIL.create();
                break;
            case "MOSSE":
                System.out.println("MOSSE case.");
                objectTracker = TrackerMOSSE.create();
                break;
            default:
                break;
        }
    }

    public void resetTracker(){
        roiRect.x = (int)trackerCoordinate.x;
        roiRect.y = (int)trackerCoordinate.y;
        roiRect.width = (int)trackerSize.x;
        roiRect.height = (int)trackerSize.y;

        roiRect2d.x = trackerCoordinate.x;
        roiRect2d.y = trackerCoordinate.y;
        roiRect2d.width = trackerSize.x;
        roiRect2d.height = trackerSize.y;


        //without value, blank rect
        //roiRect = null;
        //roiRect2d = null;
    }

    /*
    //Top view for sensoring
    private View.OnTouchListener handleDragTouch = new View.OnTouchListener() {
        @Override
        public boolean onTouch(View v, MotionEvent event) {
            int touch_x = (int) event.getX();
            int touch_y = (int) event.getY();
            Toast toast1 = Toast.makeText(MainActivity.this,
                    "X: " + touch_x + ", Y: " + touch_y, Toast.LENGTH_LONG);
            //顯示Toast
            toast1.show();
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    Log.i("TAG", "touched down");
                    break;
                case MotionEvent.ACTION_MOVE:
                    Log.i("TAG", "moving: (" + touch_x + ", " + touch_y + ")");
                    break;
                case MotionEvent.ACTION_UP:
                    Log.i("TAG", "touched up");
                    break;
            }
            return true;
        }
    };
     */

    //Media Recorder function
    public void prepareRecorder(){
        //Success for start,but shut down on stop

        //Video source is from the surface
        //Everything draw on the surface will be recorder by recorder
        recorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);

        //Store the video with time stamp
        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault());

        String currentDateandTime = sdf.format(new Date());

        recorder.setOutputFile(Environment.getExternalStorageDirectory().getAbsoluteFile()+File.separator+"/FYP/" + currentDateandTime + ".mp4");
        recorder.setVideoEncodingBitRate(1000000);
        recorder.setVideoFrameRate(60);
        recorder.setVideoSize(mOpenCvCameraView.getWidth(), mOpenCvCameraView.getHeight());
        recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        recorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);

        try {
            recorder.prepare();
            System.out.println("success prepared media recorder!!");

        } catch (IllegalStateException e) {
            Log.e("debug mediarecorder", "not prepare");
        } catch (IOException e) {
            Log.e("debug mediarecorder", "not prepare IOException");
        }

        //Initialized the mRecorder in CameraBridgeViewBase and make a new surface for recording!
        //Everything draw on that
        mOpenCvCameraView.setRecorder(recorder);

        //Only Audio
        //Success!
            /*
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
            recorder.setOutputFile(Environment.getExternalStorageDirectory().getAbsoluteFile()+File.separator+"outputAudio.3gp");
             */
    }

    public void startRecord() {
        recorder.start();
        isRecording = true;
    }

    public void stopRecord(){
        //testing add codes
        /* ignore
        recorder.setOnErrorListener(null);
        recorder.setOnInfoListener(null);
        recorder.setPreviewDisplay(null);
         */

        try{
            recorder.stop();
            isRecording = false;
        }catch(RuntimeException stopException){
            System.out.println("RuntimeException occurred!");
        }
    }




    //Set rectangle info from drag class
    public void calculateRectInfo(android.graphics.Point[] points){
            /*
            trackerCoordinate = new Point(points[0].x+40,points[0].y+20);
            trackerSize = new Point((points[2].x+40) - (points[0].x+40),(points[2].y+20) - (points[0].y+20));
            roiRect2d = new Rect2d(trackerCoordinate.x,trackerCoordinate.y,trackerSize.x,trackerSize.y);
            roiRect = new Rect((int)trackerCoordinate.x,(int)trackerCoordinate.y,(int)trackerSize.x,(int)trackerSize.y);
             */

        trackerCoordinate.x = points[0].x+40;
        trackerCoordinate.y = points[0].y+20;
        trackerSize.x = ((points[2].x+40) - (points[0].x+40));
        trackerSize.y = ((points[2].y+20) - (points[0].y+20));

        roiRect2d.x = trackerCoordinate.x;
        roiRect2d.y = trackerCoordinate.y;
        roiRect2d.width = trackerSize.x;
        roiRect2d.height =trackerSize.y;

        roiRect.x = (int)trackerCoordinate.x;
        roiRect.y =  (int)trackerCoordinate.y;
        roiRect.width = (int)trackerSize.x;
        roiRect.height = (int)trackerSize.y;

        Toast toast1 = Toast.makeText(MainActivity.this,
                "Press start to track object", Toast.LENGTH_LONG);
        //顯示Toast
        toast1.show();
    }

    //===========TensorFlowLite==========//
        //CameraActivity
    protected int[] getRgbBytes() {
        imageConverter.run();
        return rgbBytes;
    }

        //DetectorActivity
    private enum DetectorMode {
        TF_OD_API;
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    //Permission issue, solved
    /*
    @Override
    public void onRequestPermissionsResult(
            final int requestCode, final String[] permissions, final int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSIONS_REQUEST) {
            if (allPermissionsGranted(grantResults)) {
                setFragment();
            } else {
                requestPermission();
            }
        }
    }

    private static boolean allPermissionsGranted(final int[] grantResults) {
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
                Toast.makeText(
                        CameraActivity.this,
                        "Camera permission is required for this demo",
                        Toast.LENGTH_LONG)
                        .show();
            }
            requestPermissions(new String[] {PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
        }
    }
     */

    // Returns true if the device supports the required hardware level, or better.
    private boolean isHardwareLevelSupported(
            CameraCharacteristics characteristics, int requiredLevel) {
        int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            return requiredLevel == deviceLevel;
        }
        // deviceLevel is not LEGACY, can use numerical sort
        return requiredLevel <= deviceLevel;
    }

    private String chooseCamera() {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            for (final String cameraId : manager.getCameraIdList()) {
                final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

                // We don't use a front facing camera in this sample.
                final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue;
                }

                final StreamConfigurationMap map =
                        characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                if (map == null) {
                    continue;
                }

                // Fallback to camera1 API for internal cameras that don't have full support.
                // This should help with legacy situations where using the camera2 API causes
                // distorted or otherwise broken previews.
                useCamera2API =
                        (facing == CameraCharacteristics.LENS_FACING_EXTERNAL)
                                || isHardwareLevelSupported(
                                characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL);
                LOGGER.i("Camera API lv2?: %s", useCamera2API);
                return cameraId;
            }
        } catch (CameraAccessException e) {
            LOGGER.e(e, "Not allowed to access camera");
        }

        return null;
    }

    //Set Fragment, connect image to the canvas
    /*
    protected void setFragment() {
        String cameraId = chooseCamera();

        Fragment fragment;
        if (useCamera2API) {
            CameraConnectionFragment camera2Fragment =
                    CameraConnectionFragment.newInstance(
                            new CameraConnectionFragment.ConnectionCallback() {
                                @Override
                                public void onPreviewSizeChosen(final android.util.Size size, final int rotation) {
                                    previewHeight = size.getHeight();
                                    previewWidth = size.getWidth();
                                    CameraActivity.this.onPreviewSizeChosen(size, rotation);
                                }
                            },
                            this,
                            getLayoutId(),
                            getDesiredPreviewFrameSize());

            camera2Fragment.setCamera(cameraId);
            fragment = camera2Fragment;
        } else {
            fragment =
                    new LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize());
        }

        getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
    }
     */

    protected void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    public boolean isDebug() {
        return debug;
    }

    protected void readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback.run();
        }
    }

    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    /*
    //Thread function on click, ignore
    @Override
    public void onClick(View v) {
        if (v.getId() == R.id.plus) {
            String threads = threadsTextView.getText().toString().trim();
            int numThreads = Integer.parseInt(threads);
            if (numThreads >= 9) return;
            numThreads++;
            threadsTextView.setText(String.valueOf(numThreads));
            setNumThreads(numThreads);
        } else if (v.getId() == R.id.minus) {
            String threads = threadsTextView.getText().toString().trim();
            int numThreads = Integer.parseInt(threads);
            if (numThreads == 1) {
                return;
            }
            numThreads--;
            threadsTextView.setText(String.valueOf(numThreads));
            setNumThreads(numThreads);
        }
    }
     */

    /*
    protected void showFrameInfo(String frameInfo) {
        frameValueTextView.setText(frameInfo);
    }

    protected void showCropInfo(String cropInfo) {
        cropValueTextView.setText(cropInfo);
    }

    protected void showInference(String inferenceTime) {
        inferenceTimeTextView.setText(inferenceTime);
    }
     */
    protected void onPreviewSizeChosen(final android.util.Size size, final int rotation) {

        //Text Size
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());

        //BorderText
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);


        trackingOverlay.addCallback(
                new OverlayView.DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {

                        multiBoxTracker.draw(canvas);
                        if (isDebug()) {
                            multiBoxTracker.drawDebug(canvas);
                        }
                    }
                });

        multiBoxTracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    protected void processImage() {

        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();


        //將白紙鋪到畫布Canvas上
        final Canvas canvas = new Canvas(croppedBitmap);

        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        //Use worker thread to do the object detection/classification
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        //paint.setColor(Color.RED);
                        //paint.setStyle(Paint.Style.STROKE);
                        //paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();

                        //Draw Location of the rectangle, no need!!! I will draw it myself --Me
                        //Keep the logic
                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {
                                System.out.println("result.getTitle(): " + result.getTitle() + ", Location of result: " + result.getLocation() + ", .getConfidence(): " + result.getConfidence());

                                //Draw rectangle on the canvas, can be replace by our boundary

                                canvas.drawRect(location, paint);
                                //Imgproc.rectangle(mRgba, targetObjectRect ,greenColorOutline,2);

                                cropToFrameTransform.mapRect(location);

                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }

                        multiBoxTracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();

                        computingDetection = false;

                        //Display value on screen
            /*
            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });

             */
                    }
                });
    }

    //protected abstract int getLayoutId();

    protected android.util.Size getDesiredPreviewFrameSize() {
        return null;
    }

    //Pass location to Tracker


}