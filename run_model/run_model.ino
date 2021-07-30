
// Run a model
// Mathieu Guillame-Bert, 2021

#include "exported_model.h"
#include "atfdf.h"

void setup() {
  Serial.begin(9600);
}

void loop() {
  while (true) {

    // Create an example with the same value as the first examples in the training dataset.
    // The ordered list of input features is exported in the generated "exported_model.h" file.
    //
    // 0: alcohol: 9.4
    // 1: chlorides: 0.076
    // 2: citric_acid: 0.0
    // 3: density: 0.9978
    // 4: fixed_acidity: 7.4
    // 5: free_sulfur_dioxide: 11.0
    // 6: pH: 3.51
    // 7: residual_sugar: 1.9
    // 8: sulphates: 0.56
    // 9: total_sulfur_dioxide: 34.0
    // 10: volatile_acidity: 0.7
    const float example[] = {9.4f, 0.076f, 0.0f, 0.9978f,  7.4f, 11.0f, 3.51f, 1.9f, 0.56f, 34.0f, 0.7f};

    // Run the model.
    const float prediction = predict(example, kMyModel);

    Serial.print("Prediction: ");
    Serial.println(prediction, 6);

    delay(1000/*ms*/);
  }
}
