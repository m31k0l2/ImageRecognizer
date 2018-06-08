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

fun calcResult(nets: List<Network>, image: Image) = nets.map { net ->
    net.activate(image, 15.0) }.map {
        val max = it.max()!!
        it.map { if (it < max) 0.0 else 1.0 }
    }.reduce { acc, list -> acc + list }

fun main(args: Array<String>) {
    val teachNumbers = (4..6).toList()
//    val nw0123 = (1..7).map { "nets/0123/nw$it.net" }.map { CNetwork().load(it)!! }
    val nw456 = (1..11).map { "nets/456/nw$it.net" }.mapNotNull { CNetwork().load(it) }
    val batch = MNIST.buildBatch(500).filter { it.index in teachNumbers }.shuffled()
    batch.forEach {
        it.index = teachNumbers.indexOf(it.index)
    }
    var counter = 0
    batch.forEach { image ->
        val r = calcResult(nw456, image)
        val k = r.indexOf(r.max())
        val i = image.index
        if (i != k) {
            counter++
            println("${image.index} -> $k, $r")
        }

    }
    println("${1.0 - counter*1.0/batch.size}")
}

private operator fun List<Double>.plus(b: List<Double>) = zip(b).map { (a, b) -> a + b }
