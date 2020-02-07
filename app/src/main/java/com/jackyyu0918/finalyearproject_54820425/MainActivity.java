package com.jackyyu0918.finalyearproject_54820425;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
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

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";

    //private CameraBridgeViewBase mOpenCvCameraView;
    private Zoomcameraview mOpenCvCameraView;
    private boolean              mIsJavaCamera = true;
    private MenuItem             mItemSwitchCamera = null;

    //--------------------My code-----------------//
    //sensor view object
    private View SensorView;


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

    private Rect2d roiRect2d;
    private Rect roiRect;

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
    private boolean isMediaRecorderInit = false;

    //--------------------------------------------------------------------------//

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

        //zoom view setting
        mOpenCvCameraView = (Zoomcameraview)findViewById(R.id.ZoomCameraView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setZoomControl((SeekBar) findViewById(R.id.CameraZoomControls));
        mOpenCvCameraView.setCvCameraViewListener(this);

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
        trackerCoordinate = new Point(1000,200);
        trackerSize = new Point(500,200);
        greenColorOutline = new Scalar(0, 255, 0, 255);

        roiRect2d = new Rect2d(trackerCoordinate.x,trackerCoordinate.y,trackerSize.x,trackerSize.y);
        roiRect = new Rect((int)trackerCoordinate.x,(int)trackerCoordinate.y,(int)trackerSize.x,(int)trackerSize.y);

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
                createTracker(currentTrackerType);

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
                resetTraacker();
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
        SensorView = findViewById(R.id.SensorView);
        SensorView.setOnTouchListener(handleDragTouch);
    }

    @Override
    public void onPause()
    {
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
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        Size sizeRgba = mRgba.size();
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;

        //tracking section
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
            System.out.println("Tracker update result: " + objectTracker.update(mGray, roiRect2d));

            //make sure target object is inside the screen
            if(roiRect2d.x+ roiRect2d.width > 1920 || roiRect2d.x < 0 || roiRect2d.y < 0 || roiRect2d.y+ roiRect2d.height > 960 ) {
                System.out.println("Tracking Failed, target object fall outside screen");

            } else {
                //Target object matrix frame
                targetObjectMat = mRgba.submat((int)(roiRect2d.y), (int)(roiRect2d.y+ roiRect2d.height), (int)(roiRect2d.x), (int)(roiRect2d.x+ roiRect2d.width));

                if(roiRect2d.height >= roiRect2d.width) {
                    //Optimized aspect ratio for video recording (2:1)
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

        }

        //draw rectangle using Rect roiRect
        Imgproc.rectangle(mRgba,roiRect,greenColorOutline,2);

        return mRgba;
    }

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

    public void prepareRecorder(){
        //Success for start,but shut down on stop
        recorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);

        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault());

        String currentDateandTime = sdf.format(new Date());

        recorder.setOutputFile(Environment.getExternalStorageDirectory().getAbsoluteFile()+File.separator+"/FYP/" + currentDateandTime + ".mp4");
        recorder.setVideoEncodingBitRate(1000000);
        recorder.setVideoFrameRate(30);
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

    public void resetTraacker(){
        roiRect.x = (int)trackerCoordinate.x;
        roiRect.y = (int)trackerCoordinate.y;
        roiRect.width = (int)trackerSize.x;
        roiRect.height = (int)trackerSize.y;

        roiRect2d.x = trackerCoordinate.x;
        roiRect2d.y = trackerCoordinate.y;
        roiRect2d.width = trackerSize.x;
        roiRect2d.height = trackerSize.y;
    }

    //Top view for sensoring
    private View.OnTouchListener handleDragTouch = new View.OnTouchListener() {

        @Override
        public boolean onTouch(View v, MotionEvent event) {

            int x = (int) event.getX();
            int y = (int) event.getY();

            Toast toast1 = Toast.makeText(MainActivity.this,
                    "X: " + x + ", Y: " + y, Toast.LENGTH_LONG);
            //顯示Toast
            toast1.show();

            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    Log.i("TAG", "touched down");
                    break;
                case MotionEvent.ACTION_MOVE:
                    Log.i("TAG", "moving: (" + x + ", " + y + ")");
                    break;
                case MotionEvent.ACTION_UP:
                    Log.i("TAG", "touched up");
                    break;
            }

            return true;
        }
    };
}