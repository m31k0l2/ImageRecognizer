import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.*

object MNIST {
    private val mnistTrainPath = "/home/melkor/mnist_png/training"
    private val mnistTestPath = "/home/melkor/mnist_png/testing"
    const val errorPath = "error_images"

    fun buildBatch(size: Int, path: String = mnistTrainPath): Set<Image> {
        val dir = File(path)
        return (0 until size/10).flatMap {
            dir.listFiles().map {
                it.listFiles().toList().shuffled().first()
            }.map { Image(it) }
        }.toSet()
    }

    fun next(path: String = mnistTrainPath): Image {
        val files = File(path).listFiles().flatMap { it.listFiles().toList() }
        return Image(files[Random().nextInt(files.size)])
    }
}

fun calcResult(nets: List<Network>, image: Image) = nets.map { net ->
    net.activate(image, 15.0) }.map {
        val max = it.max()!!
        it.map { if (it < max) 0.0 else 1.0 }
    }.reduce { acc, list -> acc + list }

fun Image.save() {
    val dir = "${MNIST.errorPath}/$group"
    val path = "$dir/${Date().time}.png"
    File(dir).mkdir()
    Files.copy(image.toPath(), (File(path)).toPath(), StandardCopyOption.REPLACE_EXISTING)
}

fun main(args: Array<String>) {
    val teachNumbers = (7..9).toList()
//    val nw0123 = (1..7).map { "nets/0123/nw$it.net" }.map { CNetwork().load(it)!! }
    val nw789 = (1..7).map { "nets/789/union/nw$it.net" }.mapNotNull { CNetwork().load(it) }
    val batch = MNIST.buildBatch(500).filter { it.index in teachNumbers }.shuffled()
    var counter = 0
    batch.forEach { image ->
        val r = calcResult(nw789, image)
        val k = r.indexOf(r.max())
        val i = teachNumbers.indexOf(image.index)
        if (i != k) {
            counter++
            image.save()
            println("${image.index} -> $k, $r")
        }

    }
    println("${1.0 - counter*1.0/batch.size}")
}

private operator fun List<Double>.plus(b: List<Double>) = zip(b).map { (a, b) -> a + b }
