# Performace Metrics Visualization

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(hist.history['accuracy'], color='black')
plt.plot(hist.history['val_accuracy'], color='purple')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.subplot(2, 2, 2)
plt.plot(hist.history['loss'], color='black')
plt.plot(hist.history['val_loss'], color='purple')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.subplot(2, 2, 3)
plt.plot(hist.history['specificity'], color='black')
plt.plot(hist.history['val_specificity'], color='purple')
plt.title('Model Specificity')
plt.ylabel('Specificity')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.subplot(2, 2, 4)
plt.plot(hist.history['dice_coeff'], color='black')
plt.plot(hist.history['val_dice_coeff'], color='purple')
plt.title('Model dice_coeff')
plt.ylabel('dice_coeff')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.tight_layout()
plt.show()