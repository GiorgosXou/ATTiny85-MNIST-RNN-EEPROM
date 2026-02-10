/*
  Trained via Tensorflow: 22 epochs | 472 parameters (bytes)
  Testing  accuracy: 0.9216 - loss: 0.2626
  Training accuracy: 0.9186 - loss: 0.2724
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
 -9  , 21 ,-8  ,-5  , 9  ,
  17 ,-42 ,-78 ,-1  , 0  ,-4  ,-12 ,-2  ,
 -78 , 1  ,-19 ,-35 , 118, 2  , 20 ,-127,-54 ,-30 ,
};

// Pretrained int-quantized weights
const PROGMEM int8_t weights[] = {
  // Layer 0 -> 1
  114, 38 ,-100, 5  ,-4  , 11 , 21 , 17 , 21 , 27 , 29 , 39 , 32 , 69, 48 , 56 , 22 , 16 , 12 , 1  , 11 ,-4  , 2  , 24 , 5  ,-20 , 41 ,-51 ,
  58 , 21 ,-57 ,-15 ,-14 ,
 -58 ,-6  , 4  ,-11 , 45 ,-4  ,-10 ,-15 ,-28 ,-15 ,-38 ,-25 , 25 ,-5 ,-3  ,-14 , 28 , 28 , 15 , 19 ,-1  ,-4  , 6  ,-46 ,-1  , 18 , 53 , 58 ,
  79 ,-35 , 33 , 11 , 5  ,
 -126,-21 , 76 , 19 ,-90 ,-69 ,-45 ,-67 ,-65 ,-63 ,-60 ,-32 ,-68 , 2 ,-3  , 29 ,-8  ,-24 ,-13 ,-27 ,-4  ,-17 ,-14 , 10 ,-18 ,-16 , 30 ,-26 ,
  2  , 102,-30 ,-40 , 83 ,
 -65 , 1  , 16 ,-71 , 16 ,-19 , 20 , 10 , 9  , 35 , 35 , 74 , 77 , 88, 54 , 37 , 45 , 34 , 5  , 9  ,-6  ,-2  , 3  ,-11 , 9  , 62 , 118, 79 ,
 -85 ,-56 , 21 ,-3  , 18 ,
 -43 ,-18 , 8  ,-125,-92 ,-39 ,-30 ,-32 ,-16 ,-34 ,-12 ,-24 ,-51 , 6 , 11 , 43 ,-14 ,-11 ,-15 ,-20 , 0  , 5  ,-2  , 53 , 8  , 21 ,-11 ,-34 ,
  29 ,-14 ,-64 ,-42 ,-97 ,

  // Layer 1 -> 2
 -18 , 49 , 92 , 40 ,-74 ,
  37 , 116, 36 , 47 ,-43 ,-34 , 14 ,-4  ,
 -117, 6  , 110,-125, 32 ,
 -36 ,-91 , 14 ,-13 , 53 , 13 , 17 , 6  ,
  64 ,-21 , 5  ,-27 ,-127,
 -8  , 41 , 1  , 13 , 49 , 34 ,-90 ,-10 ,
 -106,-58 ,-52 ,-4  ,-27 ,
  41 , 115,-54 , 28 ,-29 , 6  ,-26 ,-35 ,
  84 ,-21 , 34 , 16 ,-82 ,
  58 ,-86 , 20 , 3  , 50 , 25 ,-42 ,-46 ,
 -44 ,-100, 28 , 28 ,-7  ,
 -77 ,-60 ,-34 ,-14 , 124, 58 , 38 , 8  ,
  5  ,-37 ,-74 ,-24 ,-64 ,
  78 , 12 , 2  ,-58 , 8  , 51 , 61 , 24 ,
  8  ,-76 , 29 , 56 , 32 ,
 -69 ,-18 , 36 ,-55 , 59 ,-49 , 29 , 94 ,

  // Layer 2 -> 3
 -62 ,-20 , 86 ,-74 ,-24 , 1  , 1  ,-15 ,
  114, 19 ,-29 , 38 , 24 , 0  , 6  ,-18 , 65 ,-33 ,
 -74 , 74 ,-29 , 34 ,-93 ,-107,-50 , 38 ,
 -2  , 109,-4  ,-30 , 14 , 53 ,-66 ,-47 , 20 , 1  ,
 -34 , 2  ,-3  , 6  , 4  , 6  ,-12 ,-18 ,
 -1  , 4  , 122,-18 ,-22 , 30 , 4  , 7  , 25 , 10 ,
 -3  , 111, 31 , 41 ,-28 , 18 , 100, 29 ,
 -30 , 15 , 44 , 44 , 31 , 53 ,-10 , 4  ,-2  ,-8  ,
  6  ,-48 ,-5  , 20 ,-51 ,-70 ,-25 , 23 ,
 -32 ,-45 , 29 , 19 , 81 ,-8  , 22 ,-15 ,-57 , 16 ,
  0  , 34 , 46 , 30 , 36 , 36 , 66 ,-10 ,
  2  ,-10 ,-42 , 32 , 15 , 87 ,-21 ,-1  ,-23 , 10 ,
 -99 , 27 ,-56 , 27 ,-5  , 23 , 72 ,-54 ,
  10 , 8  ,-4  , 3  , 8  ,-29 , 59 ,-31 , 11 , 9  ,
 -95 ,-12 , 2  ,-1  ,-49 ,-109,-10 , 24 ,
  16 ,-15 , 6  , 29 , 11 , 17 ,-63 , 69 ,-32 , 18 ,
  30 , 68 ,-14 , 91 , 29 ,-44 , 20 , 62 ,
 -29 ,-9  ,-28 , 9  ,-17 ,-6  ,-2  , 23 , 77 , 27 ,
 -48 , 79 , 38 , 18 ,-32 ,-74 , 57 ,-11 ,
  18 , 19 , 38 ,-55 , 34 , 59 ,-1  , 10 ,-63 , 76 ,
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





