#pragma once

#include "NeuroAITypes.generated.h"

UENUM(BlueprintType)
enum ENeuroActivationFunction
{
    NAct_None UMETA(DisplayName = "Linear Activation"),
    NAct_RectLinear UMETA(DisplayName = "Rectified Linear"),
    NAct_LeakyRectLinear UMETA(DisplayName = "Leaky Rectified Linear"),
    NAct_BinaryStep UMETA(DisplayName = "Binary Step"),
    NAct_Sigmoid UMETA(DisplayName = "Sigmoid"),
    NAct_TanH UMETA(DisplayName = "Hyperbolic Tangent"),
    NAct_Swish UMETA(DisplayName = "Swish"),
    NAct_ArgMax UMETA(DisplayName = "Argmax"),
    NAct_SoftMax UMETA(DisplayName = "Soft Max")
};

UENUM(BlueprintType)
enum ENeuroLossFunction
{
    NLoss_None UMETA(DisplayName = "None"),
};

// A single node in a neural network. It can have multiple inputs, but only one output.
USTRUCT(BlueprintType)
struct FNeuroNode
{
    GENERATED_BODY()
    
    // Mapping of input indices to their respective weights
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TMap<int32, float> InputWeightMap;

    // Bias added to the node output
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    float Bias;

    // When true output is clamped from -1.0 to 1.0
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    bool bClamped;

    FNeuroNode()
    {
        Bias = 0.f;
        bClamped = true;
    }

    TArray<int32> GetInputIndices() const;
    TArray<float> GetWeightsAsArray() const;

    float FeedForward(TArray<float> InputValues);
};


// A layer in a neural network. Inputs are received from the previous layer.
USTRUCT(BlueprintType)
struct FNeuroLayer
{
    GENERATED_BODY()

    // Nodes that comprise the layer
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TArray<FNeuroNode> LayerNodes;

    // Function applied to layer output
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TEnumAsByte<ENeuroActivationFunction> LayerActivationFunction;

    TArray<float> FeedForward(TArray<float> InputValues);
};

// A snapshot of a lobe's output paired to the input
USTRUCT(BlueprintType)
struct FNeuroLobeInputOutput
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TArray<float> Input;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TArray<float> Output;

    FNeuroLobeInputOutput() {};
    FNeuroLobeInputOutput(TArray<float> InputValues, TArray<float> OutputValues);
};

// A complete neural network, which itself can also be a component of larger networks
USTRUCT(BlueprintType)
struct FNeuroLobe
{
    GENERATED_BODY()

    // The neural layers that comprise the lobe
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TArray<FNeuroLayer> LobeLayers;

    // An optional way of assigning names to the inputs for human readability
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TArray<FName> InputNames;

    // An optional way of assigning names to the outputs for human readability
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TArray<FName> OutputNames;

    // Saved execution data used from computing error and applying backward propagation. NOT SAVED TO DISK
    UPROPERTY(Transient)
    TArray<FNeuroLobeInputOutput> LobeSnapshots;

    TArray<float> FeedForward(TArray<float> InputValues);

    void ClearSnapshots();
};