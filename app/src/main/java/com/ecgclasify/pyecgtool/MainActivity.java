package com.ecgclasify.pyecgtool;

import android.Manifest;
import android.animation.FloatEvaluator;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.os.Build;
import android.renderscript.Sampler;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.DragEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.jjoe64.graphview.GraphView;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.LineGraphSeries;
import com.stepstone.stepper.StepperLayout;


public class MainActivity extends AppCompatActivity {
    GraphView graph;
    float[] arreglo;
    Button carga,Visualiza,clasifica;
    String addr;
    private final int picker =1;
    PyObject func;
    PyObject object;
    Python py;
    PyObject clasificar,ecgclas;
    TextView ad_min,rest_min,value_min;
    int counter=1;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        graph=findViewById(R.id.graphic);
        carga=findViewById(R.id.boton_carga);
        Visualiza=findViewById(R.id.Visualll);
        clasifica= findViewById(R.id.clasifica);
        ad_min=findViewById(R.id.add_min);
        rest_min= findViewById(R.id.res_min);
        value_min = findViewById(R.id.val_min);
        value_min.setText(String.valueOf(counter));

        ad_min.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                counter+=1;
                value_min.setText(String.valueOf(counter));

            }
        });

        rest_min.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (counter>1) {
                    counter -= 1;
                    value_min.setText(String.valueOf(counter));
                }
            }
        });







        if (! Python.isStarted())
            Python.start(new AndroidPlatform(getApplicationContext()));


        py =Python.getInstance();



        if (ActivityCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.READ_EXTERNAL_STORAGE )!= PackageManager.PERMISSION_GRANTED){
            if (Build.VERSION.SDK_INT>=Build.VERSION_CODES.LOLLIPOP){
                ActivityCompat.requestPermissions(MainActivity.this,new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},7);
            }
        }



        carga.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                pick();


            }
        });

        Visualiza.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Cargar(addr);
                initGraph(graph);

            }
        });

        clasifica.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                clasificar=py.getModule("testmodelAndroid");
                ecgclas=clasificar.callAttr("clasificarecg");


            }
        });



    }

    void initGraph(GraphView graph) {
        graph.removeAllSeries();
        DataPoint[] points=new DataPoint[arreglo.length];
        for (int i=0;i<arreglo.length;i++){
            points[i]=new DataPoint(i,arreglo[i]);
        }

        LineGraphSeries<DataPoint> series= new LineGraphSeries<>(points);

        graph.getViewport().setXAxisBoundsManual(true);
        graph.getViewport().setMaxX(10000);
        graph.getViewport().setMinX(0);
        graph.getViewport().setMaxY(0.7);
        graph.getViewport().setMinY(-0.7);
        graph.getViewport().setScalableY(true);

        graph.getViewport().setScrollable(true);

        series.setTitle("ECG");
        graph.addSeries(series);
    }

    private void pick(){
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("file/.mat");
        intent.addCategory(Intent.CATEGORY_OPENABLE);

        try {
            startActivityForResult(
                    Intent.createChooser(intent,"Seleccione un archivo"),6);
        }
        catch (android.content.ActivityNotFoundException e){
            e.printStackTrace();

        }
    }

    protected void onActivityResult(int request,int result, Intent data){
        if (request==6){

            Log.i("entraaaaa", "entrooooooo");
            if (result== Activity.RESULT_OK)
            {
                addr=data.getData().getPath();
                Log.i("cargaaaaaa",addr);
                Toast.makeText(getApplicationContext(),addr,Toast.LENGTH_SHORT).show();

            }

            if(result==Activity.RESULT_CANCELED){

                Log.i("cancell","canceladoo");

                Toast.makeText(getApplicationContext(),"cancelado",Toast.LENGTH_SHORT).show();
            }
        }




    }


    protected void Cargar(String namedir){
        if(!namedir.equals("")){
            object= py.getModule("pyecgtool");
            func=object.callAttr("ecgsignal",namedir,counter);
            arreglo = func.toJava(float[].class);}



    }
}
