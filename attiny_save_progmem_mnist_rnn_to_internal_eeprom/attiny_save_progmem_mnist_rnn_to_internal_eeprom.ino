/*
  Trained via Tensorflow: 21 epochs | 472 parameters (bytes)
  Testing  data accuracy: 0.9160 - loss: 0.2740
  Training data accuracy: 0.9128 - loss: 0.2837
*/

#define NumberOf(arg) ((unsigned int) (sizeof (arg) / sizeof (arg [0]))) // calculates the number of layers (in this case 3)
#define _1_OPTIMIZE 0B11011010 // https://github.com/GiorgosXou/NeuralNetworks#define-macro-properties
#define _2_OPTIMIZE 0B00100100 // MULTIPLE_BIASES_PER_LAYER + int8_t quantization
#define IN_EEPROM_ADDRESS 0    // The position at which the NN will be saved at the internal EEPROM
#define Q_FLOAT_RANGE 2.0      // Our Float32 weights are in the range of [-1,1]
#define USE_RNN__NB            // Makes (vanilla)-RNN the core-architecture of your NeuralNetwork. (NB = NO_BACKPROP support)
#define SELU                   // Defines core activation-function of your NeuralNetwork.
#include <EEPROM.h>
#include <NeuralNetwork.h>


// Pretrained int-quantized biases
const PROGMEM int8_t biases[] = {
  -90 ,-24 ,-66 , 24 ,-41 ,
  -125, 34 ,-112,-40 , 110, 60 ,-95 ,-16 ,
  -126, 13 ,-108,-91 ,-8  , 83 ,-127,-41 , 118, 57
};

// Pretrained int-quantized weights
const PROGMEM int8_t weights[] = {
  // Layer 0 -> 1
   127,-125,-19 ,-48 , 59 , 59 , 48 , 50 , 50 , 64 , 64 , 64 , 75 , 73 , 60 , 46 , 50 , 56 , 33 , 26 , 24 , 19 , 24 , 18 , 30 , 25 , 46 ,-65 ,
   93 ,-85 , 58 , 23 ,-38 ,
  -27 , 117,-26 , 34 , 35 , 11 , 32 , 30 , 36 , 41 , 32 , 41 , 48 , 73 , 73 , 65 , 66 , 48 , 43 , 36 , 31 , 27 , 32 , 13 , 10 , 8  , 11 ,-127,
   31 , 67 , 37 , 57 , 64 ,
   100,-101,-121, 25 ,-84 ,-127,-127,-127,-123,-103,-107,-83 , 27 , 124, 86 , 52 , 32 , 4  , 11 , 26 , 17 ,-33 , 3  , 5  ,-1  , 24 , 112, 127,
  -48 , 20 ,-26 ,124 ,-62 ,
   13 ,-91 , 25 ,-54 ,-63 ,-88 ,-79 ,-111,-121,-121,-127,-127,-127,-103,-39 ,-26 ,-1  , 1  , 12 , 0  ,-13 , 9  ,-17 ,-38 , 9  ,-44 , 59 ,-44 ,
  -13 , 1  ,-20 , 0  ,-15 ,
  -19 ,-127, 2  , 25 ,-6  , 17 ,-1  , 10 , 20 , 31 , 37 , 49 , 14 ,-32 ,-48 ,-69 ,-86 ,-103,-105,-119,-124,-127,-127,-123,-100, 44 , 112, 118,
   14 , 10 , 56 ,-82 , 47 ,

  // Layer 1 -> 2
  -79 ,-68 , 20 , 127, 16 ,
   93 , 9  ,-73 ,-6  , 32 , 11 ,-75 ,-26 ,
  -84 ,-36 ,108 , 125, 78 ,
  -87 , 47 ,-43 , 38 ,-42 ,-27 , 19 ,-71 ,
   69 , 41 , 16 , 90 , 35 ,
   16 , 77 , 79 ,-72 ,-46 ,111 , 27 ,-69 ,
   68 ,-12 ,-2  ,-29 , 26 ,
   49 , 59 ,-60 , 75 ,-32 ,-14 , 14 ,108 ,
   106, 3  ,-11 , 20 , 87 ,
  -26 , 37 , 12 , 2  , 27 , 13 ,-15 ,-35 ,
  -55 ,-71 ,125 , 107, 47 ,
  -24 ,-33 , 14 , 3  , 70 ,-66 , 81 , 33 ,
   53 ,-8  ,-64 ,-116, 5  ,
  -12 ,-42 , 18 ,-4  ,-13 ,-34 , 74 ,-5  ,
   40 , 4  ,-120,-73 , 8  ,
  -35 ,-38 , 86 ,-16 ,-76 , 64 ,-78 , 35 ,

  // Layer 2 -> 3
  -44 ,-74 , 21 , 36 ,-78 ,-67 , 9  ,-37 ,
   74 ,-52 ,-31 , 24 ,-37 , 13 ,-39 , 8  , 32 ,-5  ,
   72 ,-26 , 65 ,-19 , 43 ,-120,-5  , 92 ,
  -26 , 110,-16 ,-7  ,-27 , 17 ,-12 ,-3  ,-63 ,-16 ,
  -8  ,-18 , 42 ,-1  , 49 , 36 ,-21 , 52 ,
  -6  , 31 ,102 ,-29 ,-3  , 7  ,-19 , 32 ,-38 ,-23 ,
  -33 ,-12 , 34 ,-86 ,-45 ,-12 , 33 ,-31 ,
  -14 ,-2  , 49 , 79 ,-5  , 33 , 36 ,-16 ,-3  , 15 ,
   41 , 35 ,-33 , 34 , 40 ,-37 , 1  ,-42 ,
  -31 , 10 , 14 ,-9  , 84 , 18 ,-2  ,-26 , 32 ,-11 ,
  -59 , 13 , 2  ,-1  ,-21 ,-15 , 21 , 10 ,
  -40 , 20 ,-29 , 37 , 33 , 65 ,-9  ,-1  , 17 , 10 ,
   14 ,-79 ,-27 , 42 , 27 ,-126, 37 , 44 ,
  -10 , 2  ,-37 ,-24 ,-27 ,-18 , 26 ,-118, 48 ,-22 ,
  -8  ,-10 ,-50 ,-12 ,-35 ,-70 ,-1  ,-66 ,
   11 , 5  ,-6  , 3  ,-10 , 54 , 1  , 83 , 10 ,-15 ,
   47 , 25 , 65 ,-47 ,-21 , 84 , 37 , 24 ,
  -44 ,-8  ,-15 ,-12 , 8  ,-4  ,-6  , 18 , 34 ,-83 ,
  -56 , 56 ,-24 ,-23 ,-14 ,-37 ,-7  , 31 ,
  -81 ,-6  , 6  ,-51 , 33 , 43 ,-7  , 36 ,-37 , 30 ,
};

const unsigned int layers[] = {28, 5, 8, 10};

void setup()
{
  delay(3000);
  pinMode(LED_BUILTIN, OUTPUT);
  NeuralNetwork NN(layers, weights, biases, NumberOf(layers));
  NN.save(IN_EEPROM_ADDRESS);
  digitalWrite(LED_BUILTIN, HIGH);
}
void loop(){}





