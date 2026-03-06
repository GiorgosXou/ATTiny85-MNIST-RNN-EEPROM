/*
  Trained via Tensorflow: 472 parameters (bytes)
  Testing  accuracy: 0.9510 - loss: 0.1618
  Training accuracy: 0.9528 - loss: 0.1528
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
 -2  , 59 ,-22 ,-99 , 35 ,
  21 ,-120,-16 , 4  ,-1  ,-38 , 3  ,-73 ,
 -127, 27 ,-60 , 78 ,-55 ,-4  ,-120,-69 ,-113,-19 ,
};

// Pretrained int-quantized weights
const PROGMEM int8_t weights[] = {
  // Layer 0 -> 1
 -127, 109, 12 , 51 , 11 , 14 , 14 , 22 , 26 , 26 , 32 , 39 , 51 , 59 , 54 , 53 , 42 , 30 , 26 , 18 , 20 , 10 , 16 , 12 , 4  ,-2  , 51 , 87 ,
  73 ,-32 , 7  , 28 ,-20 ,
  72 ,-127, 78 ,-59 ,-80 ,-97 ,-86 ,-112,-100,-98 ,-110,-105,-108,-76 ,-71 ,-57 ,-61 ,-62 ,-47 ,-48 ,-47 ,-43 ,-40 ,-57 ,-109,-57 ,-87 , 127,
  117,-24 , 30 , 80 , 9  ,
 -2  , 25 ,-24 ,-17 ,-38 , 22 ,-12 , 2  , 2  , 4  ,-1  ,-17 ,-22 ,-44 ,-23 ,-26 ,-25 , 4  ,-2  , 9  , 14 , 16 , 7  , 44 , 69 , 86 ,-16 ,-73 ,
  59 , 9  , 16 ,-36 , 47 ,
  127, 84 ,-98 ,-47 , 33 ,-69 , 15 , 17 , 0  , 24 , 43 , 52 , 59 , 51 , 36 , 14 ,-34 ,-37 ,-105,-127,-119,-93 ,-127,-117, 28 ,-17 , 7  , 127,
 -127, 77 , 48 , 75 , 94 ,
  42 , 96 , 7  , 18 , 63 , 59 , 53 , 64 , 60 , 56 , 60 , 43 , 27 ,-16 ,-14 ,-64 ,-68 ,-54 ,-56 ,-45 ,-47 ,-64 ,-10 ,-60 ,-123,-127,-105, 8  ,
  56 ,-30 ,-6  ,-22 ,-54 ,

  // Layer 1 -> 2
 -40 , 123,-72 ,-16 ,-127,
  7  , 41 ,-5  ,-63 , 66 ,-9  ,-19 ,-19 ,
 -3  ,-32 ,-5  ,-65 , 19 ,
 -76 , 106,-28 ,-78 , 52 , 15 ,-26 , 21 ,
  47 ,-127,-71 , 121,-63 ,
 -57 , 3  , 89 , 30 ,-31 ,-73 ,-26 ,-2  ,
  34 , 68 ,-111,-37 , 111,
  39 , 13 ,-17 , 45 ,-18 , 79 ,-50 ,-16 ,
  64 ,-6  ,-68 , 53 , 8  ,
  57 ,-72 , 79 , 73 ,-39 , 20 ,-45 ,-90 ,
  0  ,-126,-23 , 30 , 6  ,
  56 ,-29 ,-49 , 0  ,-19 , 58 , 35 , 5  ,
  20 , 50 ,-31 ,-36 , 12 ,
  1  ,-6  , 26 ,-26 ,-40 , 13 , 80 , 29 ,
 -17 ,-13 , 109, 25 ,-94 ,
  5  ,-68 ,-70 , 14 , 86 ,-99 ,-57 , 83 ,

  // Layer 2 -> 3
 -5  ,-25 ,-13 ,-30 ,-30 ,-42 ,-41 ,-19 ,
  112,-16 , 13 ,-25 , 12 , 34 ,-21 ,-13 , 36 ,-43 ,
  18 ,-113,-12 , 0  ,-59 ,-117,-13 ,-97 ,
 -66 , 51 ,-58 ,-82 , 0  ,-65 ,-106,-47 , 13 , 50 ,
  49 , 60 ,-22 , 59 , 34 , 104, 69 , 18 ,
 -15 ,-9  , 103,-27 , 1  ,-2  , 5  , 21 ,-4  , 18 ,
 -38 ,-8  , 40 , 7  ,-48 ,-22 , 3  ,-3  ,
  14 ,-7  , 14 , 113,-4  ,-29 ,-4  ,-7  ,-13 , 14 ,
  12 ,-8  ,-26 , 34 ,-115,-16 ,-65 ,-45 ,
  39 , 2  ,-19 , 20 , 103,-7  , 0  ,-39 , 8  , 23 ,
 -61 ,-2  , 9  , 0  , 4  ,-3  ,-31 , 2  ,
  2  , 15 ,-4  , 62 ,-16 , 99 , 14 ,-23 , 3  , 13 ,
  113,-16 , 0  , 73 , 22 , 50 ,-53 ,-11 ,
  17 ,-9  ,-47 ,-8  , 5  , 22 , 67 ,-99 , 65 , 5  ,
 -52 , 14 , 27 , 21 ,-48 ,-51 ,-9  ,-39 ,
  12 ,-11 ,-17 ,-27 , 5  , 26 ,-11 , 97 ,-15 , 7  ,
 -36 ,-11 , 41 , 26 ,-33 , 12 , 32 , 0  ,
 -2  , 5  ,-1  , 14 ,-5  ,-16 , 16 , 19 , 109, 19 ,
 -56 , 62 , 8  , 59 ,-39 ,-84 ,-29 ,-3  ,
  48 , 9  ,-21 , 31 , 5  ,-26 ,-58 ,-20 ,-11 , 109,
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





