package com.unibo.cesenamonuments;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.PopupWindow;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class MainActivity extends AppCompatActivity {

    Button camera;
    Button gallery;
    Button details;
    ImageView imageView;
    TextView result;
    final int imageSize = 224;
    List<TextView> monuments;
    float[] results;
    View popupView;
    final String[] classes = {
        "Chiesa di San Giovanni Battista",
        "Colonna dell'Ospitalità",
        "Fontana Masini",
        "Giardini Pubblici",
        "Palazzo del Ridotto",
        "Ponte Vecchio",
        "Rocca Malatestiana",
        "Teatro Bonci"
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        monuments = new ArrayList<>();
        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);
        details = findViewById(R.id.details);
        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        LayoutInflater inflater = (LayoutInflater) getSystemService(LAYOUT_INFLATER_SERVICE);
        popupView = inflater.inflate(R.layout.details, null);
        addMonuments();

        camera.setOnClickListener(view -> {
            if (checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, 3);
            } else {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
            }
        });

        gallery.setOnClickListener(view -> {
            Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(cameraIntent, 1);
        });

        details.setOnClickListener(view ->{
            if (results != null) {
                onButtonShowPopupWindowClick(view);
            }
        });
    }

    private void addMonuments() {
        monuments.add(popupView.findViewById(R.id.sangiovanni));
        monuments.add(popupView.findViewById(R.id.colonna));
        monuments.add(popupView.findViewById(R.id.fontana));
        monuments.add(popupView.findViewById(R.id.giardini));
        monuments.add(popupView.findViewById(R.id.ridotto));
        monuments.add(popupView.findViewById(R.id.ponte));
        monuments.add(popupView.findViewById(R.id.rocca));
        monuments.add(popupView.findViewById(R.id.bonci));
    }

    private void onButtonShowPopupWindowClick(View view) {
        for(int i = 0; i < monuments.size(); i++) {
            String text = classes[i] + ": ";
            float value = results[i];
            monuments.get(i).setText(text + (value < 0.01 ? "<0.01%" : Math.floor(value * 100) / 100 + "%"));
        }

        int width = LinearLayout.LayoutParams.WRAP_CONTENT;
        int height = LinearLayout.LayoutParams.WRAP_CONTENT;
        boolean focusable = true;
        final PopupWindow popupWindow = new PopupWindow(popupView, width, height, focusable);
        popupWindow.showAtLocation(view, Gravity.CENTER, 0, 0);
        popupView.setOnTouchListener((v, event) -> {
            popupWindow.dismiss();
            return true;
        });
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor assetFileDescriptor = this.getAssets().openFd("model.tflite");
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = assetFileDescriptor.getStartOffset();
        long len = assetFileDescriptor.getLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, len);
    }

    private static int getMaxPosition(final float[] in) {
        float maxVal = 0.0f;
        int maxPos = 0;
        for (int i = 0; i < in.length; i++) {
            if (in[i] > maxVal) {
                maxVal = in[i];
                maxPos = i;
            }
        }
        return maxPos;
    }

    private void classifyImage(Bitmap image) {
        ImageProcessor imageProcessor =
            new ImageProcessor.Builder()
                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(127.5f, 127.5f))
                .build();
        TensorImage tImage = new TensorImage(DataType.FLOAT32);
        tImage.load(image);
        tImage = imageProcessor.process(tImage);
        TensorBuffer probabilityBuffer = TensorBuffer.createFixedSize(new int[]{1, 8}, DataType.FLOAT32);
        try {
            new Interpreter(loadModelFile()).run(tImage.getBuffer(), probabilityBuffer.getBuffer());
            int maxPos = getMaxPosition(probabilityBuffer.getFloatArray());
            result.setText(classes[maxPos]);
            results = probabilityBuffer.getFloatArray();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){
            if(requestCode == 3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            } else {
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}