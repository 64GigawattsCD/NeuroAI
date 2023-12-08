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
enum ENeuroErrorFunction
{
    NErr_None UMETA(DisplayName = "Linear Activation"),
    NErr_RectLinear UMETA(DisplayName = "Rectified Linear"),
    NErr_LeakyRectLinear UMETA(DisplayName = "Leaky Rectified Linear"),
    NErr_BinaryStep UMETA(DisplayName = "Binary Step"),
    NErr_Sigmoid UMETA(DisplayName = "Sigmoid"),
    NErr_TanH UMETA(DisplayName = "Hyperbolic Tangent"),
    NErr_Swish UMETA(DisplayName = "Swish"),
    NErr_ArgMax UMETA(DisplayName = "Argmax"),
    NErr_SoftMax UMETA(DisplayName = "Soft Max")
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

// A snapshot of a lobe's output paired to the input and error list
USTRUCT(BlueprintType)
struct FNeuroLobeInputOutput
{
    GENERATED_BODY()

    // Snapshots of inputs passed in
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TArray<float> Input;

    // Snapshots of output
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TArray<float> Output;

    // Tracking desired output per each input set
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TArray<float> DesiredOutput;

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
    void SetDesiredOutputs(TArray<float> InOutputs, int32 Index = -1);

    void ClearSnapshots();
};

// A generation of neuro lobes paired with their fitness scores
USTRUCT(BlueprintType)
struct FNeuroGeneration
{
    GENERATED_BODY()
    
    // 
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TArray<FNeuroLobe> GenerationLobes;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TArray<float> GenerationScores;
    
    void SetGenerationLobes(const TArray<FNeuroLobe> InLobes);
    void SetGenerationScores(const TArray<float> InScores);
    bool GetHighestScoringLobe(FNeuroLobe& OutLobe);
};

// An entire lineage of generations of lobes
USTRUCT(BlueprintType)
struct FNeuroLineage
{
    GENERATED_BODY()
    
    // Mapping of individual lobes paired with fitness scores
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Neuron")
    TArray<FNeuroGeneration> LineageGenerations;

    void AppendGeneration(FNeuroGeneration & InGeneration);
    void AppendGenerations(TArray<FNeuroGeneration> InGenerations);
    FNeuroGeneration GetLatestGeneration();
};

template <typename InElementType>
struct TPriorityQueueNode {
    InElementType Element;
    float Priority;

    TPriorityQueueNode()
    {
    }

    TPriorityQueueNode(InElementType InElement, float InPriority)
    {
        Element = InElement;
        Priority = InPriority;
    }

    bool operator<(const TPriorityQueueNode<InElementType> Other) const
    {
        return Priority < Other.Priority;
    }
};

template <typename InElementType>
class TPriorityQueue {
public:
    TPriorityQueue()
    {
        Array.Heapify();
    }

public:
    // Always check if IsEmpty() before Pop-ing!
    InElementType Pop()
    {
        TPriorityQueueNode<InElementType> Node;
        Array.HeapPop(Node);
        return Node.Element;
    }

    TPriorityQueueNode<InElementType> PopNode()
    {
        TPriorityQueueNode<InElementType> Node;
        Array.HeapPop(Node);
        return Node;
    }

    void Push(InElementType Element, float Priority)
    {
        Array.HeapPush(TPriorityQueueNode<InElementType>(Element, Priority));
    }

    bool IsEmpty() const
    {
        return Array.Num() == 0;
    }

private:
    TArray<TPriorityQueueNode<InElementType>> Array;
};