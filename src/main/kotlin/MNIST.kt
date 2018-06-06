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
    val teachNumbers = (0..3).toList()
    val nw1 = CNetwork().load("nets/0123/nw1.net")!!
    val nw2 = CNetwork().load("nets/0123/nw2.net")!!
    val nw3 = CNetwork().load("nets/0123/nw3.net")!!
    val batch = MNIST.buildBatch(500).filter { it.index in teachNumbers }.shuffled()
    var counter = 0
    batch.forEach {
        val r1 = nw1.activate(it, 15.0)
        val r2 = nw2.activate(it, 15.0)
        val r3 = nw3.activate(it, 15.0)
        val r = r1 + r2 + r3
        val k = r.indexOf(r.max())
        val i = it.index
        if (i != k) counter++
        println("${it.index} -> $k")
    }
    println("${1.0 - counter*1.0/batch.size}")
}

private operator fun List<Double>.plus(b: List<Double>) = zip(b).map { (a, b) -> a + b }
