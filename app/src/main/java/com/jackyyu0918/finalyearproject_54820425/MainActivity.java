package com.jackyyu0918.finalyearproject_54820425;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
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
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.widget.SwitchCompat;
import androidx.core.app.ActivityCompat;
import androidx.preference.PreferenceManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;

//TensorflowLite library
import com.jackyyu0918.finalyearproject_54820425.customview.OverlayView;
import com.jackyyu0918.finalyearproject_54820425.tflite.*;
import com.jackyyu0918.finalyearproject_54820425.env.*;
import com.jackyyu0918.finalyearproject_54820425.tracking.MultiBoxTracker;

//Media Recorder


public class MainActivity extends Activity implements CvCameraViewListener2, View.OnClickListener {
    private static final String TAG = "OCVSample::Activity";

    //private CameraBridgeViewBase mOpenCvCameraView;
    private Zoomcameraview mOpenCvCameraView;
    private boolean mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;

    //--------------------Class Field-----------------//
    //Permission
    int PERMISSION_ALL = 1;

    //==============Camera info====================//
    private Camera camera;
    private byte[] bytes;

    //==============Tracker variable===============//
    //Matrix
    private Mat mRgba;
    private Mat mGray;
    private Mat testMat;

    private Mat targetObjectMat;
    private Mat zoomWindowMat;
    private Mat optimizeObjectMat;

    private Rect2d targetObjectRect2d = null;
    private double targetObject_x;
    private double targetObject_y;
    private double targetObject_width;
    private double targetObject_height;

    //Tracker
    //===============Tracker=================//
    private TrackingHandler trackingHandler;
    private boolean isInitTracker = false;
    private int borderThicknessInt = 2;


    //Mode switching
    //false = small window
    private boolean isFullScreen = false;

    //===============View===============//
    //button
    // for init tracker
    private ImageButton button_startTrack;

    // for reset tracker
    private Button button_resetTrack;

    // for full view tracking
    private Button button_fullView;

    // for start recording
    private ImageButton button_startRecord;

    // for pausing object detection
    private ImageButton button_pauseObjectDetection;

    // for going setting page
    private ImageButton button_setting;

    //View object
    private DragRegionView DragRegionView;
    OverlayView trackingOverlay;

    private boolean isDisplayInferenceTime = false;
    private boolean isDisplayThread = false;
    private RelativeLayout threadlayout;
    private LinearLayout inferenceTimeLayout;

    //Switch
    // for switching manual mode and object detection mode
    private SwitchCompat modeSwitch;

    //Spinner for object detected
    Spinner detectedObjectSpinner = null;

    private ImageView plusImageView, minusImageView;

    private TextView threadsTextView;
    protected TextView inferenceTimeTextView;

    //===============Media Recorder=====================//
    private RecordingHandler RecordingHandler;
    private boolean isRecording = false;

    //True  = Object detection mode, auto detect object
    //False = ManualMode, user drag out the boundary
    private boolean objectDetectionFeature = false;

    //=============TensorFlowLite variable==============//
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

    //Model and Label
    private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;

    // Minimum detection confidence to track a detection. 0.5/50%
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;

    private static final boolean MAINTAIN_ASPECT = false;

    //private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final android.util.Size DESIRED_PREVIEW_SIZE = new android.util.Size(1920, 960);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    //Sensor Orientation
    private Integer sensorOrientation;

    //Classifier creates tflite, which is the interpreter
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

    private boolean debug = false;

    //Drop down menu variable
    private ArrayAdapter<String> detectedObjectNameAdaptar = null;

    //TensorFlow MultiBoxTracker
    private MultiBoxTracker multiBoxTracker;

    //Tracking result
    private List<Classifier.Recognition> RecognizedItemList = null;
    private Classifier.Recognition selectedRecognizedItem = null;

    //----------------------------end of class field----------------------------------//

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
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
        mOpenCvCameraView = (Zoomcameraview) findViewById(R.id.ZoomCameraView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setZoomControl((SeekBar) findViewById(R.id.CameraZoomControls));
        mOpenCvCameraView.setCvCameraViewListener(this);

        //Grant permission
        String[] PERMISSIONS = {
                android.Manifest.permission.CAMERA,
                android.Manifest.permission.WRITE_EXTERNAL_STORAGE,
                android.Manifest.permission.RECORD_AUDIO,
        };

        if (!hasPermissions(this, PERMISSIONS)) {
            ActivityCompat.requestPermissions(this, PERMISSIONS, PERMISSION_ALL);
        }

        //Load Preference (Setting)
        PreferenceManager.setDefaultValues(this, R.xml.root_preferences, false);

        //======Tracker section=======//
        trackingHandler = TrackingHandler.getInstance();

        //========End of Tracker section=========//

        //===========Media Recorder section========//
        //Singleton handler
        RecordingHandler = RecordingHandler.getInstance();

        //View
        //start button
        button_startTrack = findViewById(R.id.button_startTrack);
        button_startTrack.setOnClickListener(this);

        //switch mode button
        button_fullView = findViewById(R.id.button_fullView);
        button_fullView.setOnClickListener(this);

        //reset button
        button_resetTrack = findViewById(R.id.button_resetTracker);
        button_resetTrack.setOnClickListener(this);

        //start recording button
        button_startRecord = findViewById(R.id.button_startRecord);
        button_startRecord.setOnClickListener(this);
        //Long click detection -- Recording!

        button_startRecord.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {
                // prepareRecord();  <-- Moved to onCreate()
                if (isRecording == false) {
                    RecordingHandler.prepareRecorder(mOpenCvCameraView);

                    //Recording message
                    Toast toast = Toast.makeText(MainActivity.this,
                            "Start recording...", Toast.LENGTH_LONG);
                    //顯示Toast
                    toast.show();
                    //button_startRecord.setText("STOP RECORDING");
                    RecordingHandler.startRecord(isRecording);

                    button_startRecord.setImageResource(R.drawable.button_stoprecording);
                    isRecording = true;
                }
                return true;
            }
        });


        //pause object detection button
        button_pauseObjectDetection = findViewById(R.id.button_pauseObjectDetection);
        button_pauseObjectDetection.setOnClickListener(this);

        //setting button
        button_setting = findViewById(R.id.button_setting);
        button_setting.setOnClickListener(this);

        //spinner tracker selection
        detectedObjectSpinner = findViewById(R.id.detectedObjectSpinner);

        //button for adding/minusing thread
        threadsTextView = findViewById(R.id.threads);
        plusImageView = findViewById(R.id.plus);
        minusImageView = findViewById(R.id.minus);

        plusImageView.setOnClickListener(this);
        minusImageView.setOnClickListener(this);

        //final ArrayAdapter<String> nameAdaptar = new ArrayAdapter<String>(MainActivity.this,android.R.layout.simple_expandable_list_item_1, getResources().getStringArray(R.array.trackingAlgorithmName));
        //nameAdaptar.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        //trackerSpinner.setAdapter(nameAdaptar);

        //Thread related view
        threadlayout = findViewById(R.id.threadLayout);
        inferenceTimeLayout = findViewById(R.id.inferenceTimeLayout);
        inferenceTimeTextView = findViewById(R.id.inference_info);

        final List<String> detectedItemList = new ArrayList<String>();

        detectedObjectNameAdaptar = new ArrayAdapter<String>(MainActivity.this, android.R.layout.simple_expandable_list_item_1, detectedItemList);
        detectedObjectNameAdaptar.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        detectedObjectSpinner.setAdapter(detectedObjectNameAdaptar);

        //onSelectItem
        detectedObjectSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                Toast.makeText(MainActivity.this, "You are choosing " + parent.getSelectedItem().toString() + ".", Toast.LENGTH_SHORT).show();

                //Already exist selected object, choose another one will trigger reset
                if (selectedRecognizedItem != null) {
                    isInitTracker = false;
                    trackingHandler.setTracker(null);
                    trackingHandler.resetTrackerDetails();
                }

                if (RecognizedItemList != null && position != 0) {

                    selectedRecognizedItem = RecognizedItemList.get(position - 1);
                    System.out.println("selectedRecognizedItem: " + selectedRecognizedItem.getTitle() + ", Location: " + selectedRecognizedItem.getLocation() + ", Confidence: " + selectedRecognizedItem.getConfidence() + " is selected in the drop-down menu!");

                    //Set view disable
                    trackingOverlay.setVisibility(View.GONE);

                    //stop object tracking
                    objectDetectionFeature = false;
                    trackingHandler.resetTrackerDetails();
                    trackingHandler.createTracker(getValueFromPerference("trackertype", MainActivity.this));

                    //get user drag result
                    trackingHandler.setTrackerSize((int) selectedRecognizedItem.getLocation().left, (int) selectedRecognizedItem.getLocation().top, (int) (selectedRecognizedItem.getLocation().right - selectedRecognizedItem.getLocation().left), (int) (selectedRecognizedItem.getLocation().bottom - selectedRecognizedItem.getLocation().top));

                    //tracker initialization
                    trackingHandler.initializeTracker(mGray);
                    isInitTracker = true;
                    //System.out.println("Tracker init result: " + firstTracker.init(mGray,roiRect2d));

                    //Tracker message
                    Toast toast1 = Toast.makeText(MainActivity.this,
                            "Current tracker: " + trackingHandler.getTracker().getClass(), Toast.LENGTH_LONG);
                    //顯示Toast
                    toast1.show();

                    Toast toast2 = Toast.makeText(MainActivity.this,
                            "Current camera size: " + mOpenCvCameraView.getWidth() + "x" + mOpenCvCameraView.getHeight(), Toast.LENGTH_LONG);
                    //顯示Toast
                    toast2.show();

                    Toast toast3 = Toast.makeText(MainActivity.this,
                            "Current tracker size: " + trackingHandler.getroiRect().width + "x" + trackingHandler.getroiRect().height, Toast.LENGTH_LONG);
                    //顯示Toast
                    toast3.show();
                } else {
                    Toast.makeText(MainActivity.this, "Please select detected object", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                Toast.makeText(MainActivity.this, "Nothing is selected.", Toast.LENGTH_LONG).show();
            }
        });


        //=======End of Button Configuration==========//

        //Sensor View at top
        DragRegionView = findViewById(R.id.SensorView);

        //Switch
        modeSwitch = findViewById(R.id.modeSwitch);
        modeSwitch.setOnClickListener(this);

        //Overlay View for object tracking
        //For drawing rectangle
        trackingOverlay = findViewById(R.id.tracking_overlay);
    }

    @Override
    public void onPause() {
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
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        //Multi Thread issue
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
        //testMat = new Mat();
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
        //=======Basic Info======//
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        Size sizeRgba = mRgba.size();
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;
        //=======End of Basic Info=======//

        //================Object detection================//
        //CameraActivity
        if (objectDetectionFeature == true) {
            System.out.println("Performing object tracking!!.....");

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
        }
        //===============End of object tracking section==============//


        //============Standard Tracking section======================//
        // if initialized tracker, start update the ROI
        if (isInitTracker) {
            //Pre-defined target window details: x,y,width,height
            //Assign 2d to 1d:
            // 2d: update by tracker
            // 1d: update Rec
            trackingHandler.sysnRectValue();

            //Update tracker information to roiRect2d
            //Why not multi thread, high latency
            System.out.println("Tracker update result: " + trackingHandler.updateTracker(mGray));

            //make sure target object is inside the screen

            //Target object matrix frame
            targetObjectMat = mRgba.submat((int) (trackingHandler.getRoiRect2d().y), (int) (trackingHandler.getRoiRect2d().y + trackingHandler.getRoiRect2d().height), (int) (trackingHandler.getRoiRect2d().x), (int) (trackingHandler.getRoiRect2d().x + trackingHandler.getRoiRect2d().width));

            targetObjectRect2d = trackingHandler.getRoiRect2d();


            targetObject_x = trackingHandler.getRoiRect2d().x;
            targetObject_y = trackingHandler.getRoiRect2d().y;
            targetObject_width = trackingHandler.getRoiRect2d().width;
            targetObject_height = trackingHandler.getRoiRect2d().height;

            if (targetObject_height >= targetObject_width) {
                //Optimized aspect ratio for video recording (2:1)
                optimizeObjectMat = mRgba.submat((int) (targetObject_y), (int) (targetObject_y + targetObject_height), (int) (trackingHandler.getRoiRect2d().x + (targetObject_width / 2) - targetObject_height), (int) (trackingHandler.getRoiRect2d().x + (targetObject_width / 2) + targetObject_height));
            } else {
                //Optimized aspect ratio for video recording (2:1)
                optimizeObjectMat = mRgba.submat((int) (targetObject_y + (targetObject_height / 2) - targetObject_width / 4), (int) ((targetObject_y + (targetObject_height / 2) - targetObject_width / 4) + targetObject_width / 2), (int) (trackingHandler.getRoiRect2d().x), (int) (trackingHandler.getRoiRect2d().x + targetObject_width));
            }


            // Small window preview mode
            if (isFullScreen == false) {
                //top-left corner of mRgba
                zoomWindowMat = mRgba.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);

                //show target matrix at the top-left corner
                //Imgproc.resize(targetObjectMat, zoomWindowMat, zoomWindowMat.size(), 0, 0, Imgproc.INTER_LINEAR_EXACT);
                Imgproc.resize(optimizeObjectMat, zoomWindowMat, zoomWindowMat.size(), 0, 0, Imgproc.INTER_LINEAR_EXACT);

                // Full screen mode (for video recording)
            } else {
                System.out.print("test Mat" + testMat);
                //mRgba.copyTo(testMat);
                testMat = mRgba.clone();

                //full the screen with target matrix
                //Imgproc.resize(targetObjectMat, testMat, mRgba.size(), 0, 0, Imgproc.INTER_LINEAR_EXACT);
                Imgproc.resize(optimizeObjectMat, testMat, mRgba.size(), 0, 0, Imgproc.INTER_LINEAR_EXACT);
                //draw on full screen

                Imgproc.rectangle(testMat, trackingHandler.getroiRect(), trackingHandler.getGreenColorOutline(), borderThicknessInt);
                return testMat;
            }


            Imgproc.rectangle(mRgba, trackingHandler.getroiRect(), trackingHandler.getGreenColorOutline(), borderThicknessInt);
            if (selectedRecognizedItem != null) {
                Imgproc.putText(mRgba, selectedRecognizedItem.getTitle(), new Point(trackingHandler.getroiRect().x, trackingHandler.getroiRect().y), 1, 5, trackingHandler.getGreenColorOutline());
            } else {
                Imgproc.putText(mRgba, "Target Object", new Point(trackingHandler.getroiRect().x, trackingHandler.getroiRect().y), 1, 5, trackingHandler.getGreenColorOutline());
            }
        }


        //System.out.println("isDisplayInference; " + isDisplayInferenceTime);
        //System.out.println("Get preference valie: " + getValueFromPerference("inferencetime", MainActivity.this));
        //System.out.println("isDisplayInference: " + isDisplayThread);

        return mRgba;
        //============End of normal tracking==============//
    }

    //================End of onCameraPreview=========//

    //Tracker function
    //my own function
    //===========Object detection functiion==========//
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

    protected void showInference(String inferenceTime) {
        inferenceTimeTextView.setText(inferenceTime);
    }

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

        //Call worker thread to do the object detection
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
                        RecognizedItemList = mappedRecognitions;
                        trackingOverlay.postInvalidate();

                        computingDetection = false;

                        //Display value on screen
                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });
                    }
                });
    }

    //protected abstract int getLayoutId();

    protected android.util.Size getDesiredPreviewFrameSize() {
        return null;
    }

    protected void setNumThreads(final int numThreads) {
        runInBackground(new Runnable() {
            @Override
            public void run() {
                detector.setNumThreads(numThreads);
            }
        });
    }

    //=====================end of object detection section=================//

    //Stop object detection feature
    //Can pause the object detection action
    protected void stopObjectDetection() {

        //Make sure the object detector detected some object
        if (RecognizedItemList != null) {

            //terminate worker thread to stop further detection
            handlerThread.quitSafely();
            try {
                handlerThread.join();
                handlerThread = null;
                handler = null;
            } catch (final InterruptedException e) {
                System.out.println(e);
            }

            objectDetectionFeature = false;

            //Get the List of Recognized Item
            System.out.println("RecognizedItemList.size(): " + RecognizedItemList.size());
            updateSpinnerMenu();
        }

    }

    protected void updateSpinnerMenu() {
        //Add item to drop down list and refresh
        detectedObjectNameAdaptar.clear();
        detectedObjectNameAdaptar.add("Please select detected object");
        for (Classifier.Recognition result : RecognizedItemList) {
            detectedObjectNameAdaptar.add(result.getTitle() + " " + result.getConfidence() * 100 + "%");

            System.out.println("Back Tracking: " + result.getTitle() + ", " + result.getLocation());
        }
        detectedObjectNameAdaptar.notifyDataSetChanged();
    }

    //Get value from shared preference (Setting)
    public static String getValueFromPerference(String key, Context context) {
        SharedPreferences preferences = PreferenceManager.getDefaultSharedPreferences(context);
        return preferences.getString(key, null);
    }

    //Centralizing button on click event
    @Override
    public void onClick(View v) {
        switch (v.getId()) {

            case R.id.button_startTrack:
                if (!objectDetectionFeature) {

                    //own function, create proper tracker
                    if (DragRegionView.points[0] == null) {
                        Toast toast1 = Toast.makeText(MainActivity.this,
                                "Please drag on target object.", Toast.LENGTH_LONG);
                        //顯示Toast
                        toast1.show();
                    } else {
                        //Dynamic tracker Will be implemented in setting
                        trackingHandler.createTracker(getValueFromPerference("trackertype", MainActivity.this));
                        //createTracker("MOSSE");


                        //get user drag result
                        trackingHandler.calculateRectInfo(DragRegionView.points);

                        //tracker initialization
                        trackingHandler.initializeTracker(mGray);
                        //System.out.println("Tracker init result: " + firstTracker.init(mGray,roiRect2d));
                        isInitTracker = true;

                        //Tracker message
                        Toast toast1 = Toast.makeText(MainActivity.this,
                                "Current tracker: " + trackingHandler.getTracker().getClass() + ", camera size: \" + mOpenCvCameraView.getWidth() + \"x\" + mOpenCvCameraView.getHeight()", Toast.LENGTH_LONG);
                        //顯示Toast
                        toast1.show();

                        DragRegionView.isReset = true;
                        DragRegionView.invalidate();

                        Toast toast2 = Toast.makeText(MainActivity.this,
                                "Current tracker size: " + trackingHandler.getroiRect().width + "x" + trackingHandler.getroiRect().height, Toast.LENGTH_LONG);
                        //顯示Toast
                        toast2.show();
                    }
                }
                break;

            case R.id.button_fullView:
                if (isFullScreen == false) {
                    isFullScreen = true;
                } else {
                    isFullScreen = false;
                }
                //Tracker message
                Toast toast = Toast.makeText(MainActivity.this,
                        "Full screen mode: " + isFullScreen, Toast.LENGTH_LONG);
                //顯示Toast
                toast.show();
                break;

            case R.id.button_resetTracker:
                isInitTracker = false;
                trackingHandler.setTracker(null);
                trackingHandler.resetTrackerDetails();

                DragRegionView.isReset = true;
                DragRegionView.invalidate();
                //testing****
                //System.out.println("Coordinate: " + DragRegionView.points[0] + DragRegionView.points[1] + DragRegionView.points[2] + DragRegionView.points[3]);

                //Automatic mode?
                if(objectDetectionFeature == false && modeSwitch.isChecked()){
                    objectDetectionFeature = true;
                    rgbBytes = null;
                }
                break;

            case R.id.button_startRecord:
                //Photo shooting/ScreenShot
                if (isRecording == false) {
                    Mat captureMatrix = null;
                    if (isInitTracker) {
                        captureMatrix = testMat;
                    } else {
                        captureMatrix = mRgba;
                    }

                    Bitmap bmp = Bitmap.createBitmap(captureMatrix.width(), captureMatrix.height(), Bitmap.Config.ARGB_8888);
                    Mat tmp = new Mat(captureMatrix.width(), captureMatrix.height(), CvType.CV_8UC1, new Scalar(4));
                    System.out.println("tmp: " + tmp + ", bmp: " + bmp);

                    //converting matrix to Bmp
                    try {
                        Imgproc.cvtColor(captureMatrix, tmp, Imgproc.COLOR_GRAY2RGBA, 4);
                        Utils.matToBitmap(captureMatrix, bmp);
                    } catch (CvException e) {
                        Log.d("Exception", e.getMessage());
                    }

                    //Store bitmap to local storage
                    try (FileOutputStream out = new FileOutputStream(Environment.getExternalStorageDirectory().getAbsoluteFile() + File.separator + "/FYP/" + "image_" + generateDateInfo() + ".bmp")) {
                        bmp.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
                        Toast toast1 = Toast.makeText(MainActivity.this,
                                "ScreenShot saved as " + Environment.getExternalStorageDirectory().getAbsoluteFile() + File.separator + "/FYP/" + "image_" + generateDateInfo() + ".bmp", Toast.LENGTH_LONG);
                        //顯示Toast
                        toast1.show();
                        // PNG is a lossless format, the compression factor (100) is ignored
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    //Is recording
                } else {
                    //Finish video recording if is recording
                    //Recording message
                    Toast toast2 = Toast.makeText(MainActivity.this,
                            "End recording...", Toast.LENGTH_LONG);
                    //顯示Toast
                    toast2.show();
                    //button_startRecord.setText("START RECORDING");

                    //empty the recorder before stop
                    mOpenCvCameraView.setRecorder(null);
                    RecordingHandler.stopRecord();
                    button_startRecord.setImageResource(R.drawable.button_startrecording);
                    isRecording = false;
                }
                break;

            case R.id.button_pauseObjectDetection:
                //testing button
                if (objectDetectionFeature == true) {
                    stopObjectDetection();
                    //button change to continue
                    button_pauseObjectDetection.setImageResource(R.drawable.button_continue_50x50);
                } else {
                    //
                    objectDetectionFeature = true;
                    button_pauseObjectDetection.setImageResource(R.drawable.button_pause_50x50);
                }
                break;

            case R.id.button_setting:
                //testing button
                Intent settingPageIntent = new Intent(v.getContext(), SettingsActivity.class);
                startActivity(settingPageIntent);
                break;

            case R.id.modeSwitch:
                //isChecked = clicked/ball move to the right
                //Turn on object detection feature
                if (modeSwitch.isChecked()) {
                    trackingHandler.resetTrackerDetails();
                    objectDetectionFeature = true;

                    //testing
                    testMat = null;
                    optimizeObjectMat = null;

                    //Display/Hide unused button/view
                    trackingOverlay.setVisibility(View.VISIBLE);
                    button_pauseObjectDetection.setVisibility(View.VISIBLE);
                    detectedObjectSpinner.setVisibility(View.VISIBLE);

                    updateInferenceAndThreadStatus();

                    DragRegionView.setVisibility(View.GONE);
                    button_startTrack.setVisibility(View.GONE);

                    //button_resetTrack.setVisibility(View.GONE);

                    Toast toast1 = Toast.makeText(MainActivity.this,
                            "Object Detection mode has turned ON.", Toast.LENGTH_LONG);
                    //顯示Toast
                    toast1.show();
                } else {
                    trackingHandler.resetTrackerDetails();
                    objectDetectionFeature = false;

                    //Display/Hide unused button/view
                    trackingOverlay.setVisibility(View.GONE);
                    button_pauseObjectDetection.setVisibility(View.GONE);
                    detectedObjectSpinner.setVisibility(View.GONE);
                    threadlayout.setVisibility(View.GONE);
                    inferenceTimeLayout.setVisibility(View.GONE);

                    DragRegionView.setVisibility(View.VISIBLE);
                    button_startTrack.setVisibility(View.VISIBLE);
                    button_resetTrack.setVisibility(View.VISIBLE);

                    DragRegionView.setEnabled(true);

                    Toast toast1 = Toast.makeText(MainActivity.this,
                            "Object Detection mode has turned OFF.", Toast.LENGTH_LONG);
                    //顯示Toast
                    toast1.show();
                }
                break;

            case R.id.plus:
                String threads = threadsTextView.getText().toString().trim();
                int numThreads = Integer.parseInt(threads);
                if (numThreads >= 9) return;
                numThreads++;
                threadsTextView.setText(String.valueOf(numThreads));
                setNumThreads(numThreads);
                break;

            case R.id.minus:
                threads = threadsTextView.getText().toString().trim();
                numThreads = Integer.parseInt(threads);
                if (numThreads == 1) {
                    return;
                }
                numThreads--;
                threadsTextView.setText(String.valueOf(numThreads));
                setNumThreads(numThreads);
                break;
            }
        }

    //Grant permission for Camera, Audio and ExternalStorageAccess
    public static boolean hasPermissions(Context context, String... permissions) {
        if (context != null && permissions != null) {
            for (String permission : permissions) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    return false;
                }
            }
        }
        return true;
    }

    //Generate time stamp for output file
    public String generateDateInfo() {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault());
        String currentDateandTime = sdf.format(new Date());
        return currentDateandTime;
    }

    public void updateInferenceAndThreadStatus(){
        //Null object!
        if(getValueFromPerference("inferencetime", MainActivity.this).equals("Display")){
            isDisplayInferenceTime = true;
            if(objectDetectionFeature) inferenceTimeLayout.setVisibility(View.VISIBLE);
        }
        else isDisplayInferenceTime = false;

        if(getValueFromPerference("threaddisplay", MainActivity.this).equals("Display")){
            isDisplayThread = true;
            if(objectDetectionFeature) threadlayout.setVisibility(View.VISIBLE);
        }
        else isDisplayThread = false;
    }
}