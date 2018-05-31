import java.io.File
import java.util.*

object MNIST {
    private val mnistTrainPath = "/home/melkor/mnist_png/training"
    private val mnistTestPath = "/home/melkor/mnist_png/testing"
    private val dir = File(mnistTrainPath)

    fun buildBatch(size: Int): List<Image> {
        return (0 until size/10).flatMap {
            dir.listFiles().map {
                it.listFiles().toList().shuffled().first()
            }.map { Image(it) }
        }
    }

    fun next(): Image {
        val files = File(mnistTestPath).listFiles().flatMap { it.listFiles().toList() }
        return Image(files[Random().nextInt(files.size)])
    }
}

fun main(args: Array<String>) {
    val teachNumbers = (0..9).toList()
    val nw = NetworkIO().load("nets/nw.net")!!
    val batch = MNIST.buildBatch(500).filter { it.index in teachNumbers }.shuffled()
    var counter = 0
    Neuron.alpha = 15.0
    batch.forEach {
        val r = nw.activate(it)
        val k = r.indexOf(r.max())
        val i = it.index
        if (i != k) counter++
        println("${it.index} -> $k")
    }
    println("${1.0 - counter*1.0/batch.size}")
}