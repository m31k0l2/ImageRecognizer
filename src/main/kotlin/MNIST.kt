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

    fun allSet(path: String = mnistTrainPath): Set<Image> {
        val dir = File(path)
        return dir.listFiles().flatMap {
            it.listFiles().map { Image(it) }
        }.toSet()
    }

    fun next(path: String = mnistTrainPath): Image {
        val files = File(path).listFiles().flatMap { it.listFiles().toList() }
        return Image(files[Random().nextInt(files.size)])
    }
}

fun Image.save() {
    val dir = "${MNIST.errorPath}/$group"
    val path = "$dir/$name"
    File(dir).mkdir()
    Files.copy(image.toPath(), (File(path)).toPath(), StandardCopyOption.REPLACE_EXISTING)
}

fun Image.delete() {
    val dir = "${MNIST.errorPath}/$group"
    File("$dir/$name").delete()
}

fun main(args: Array<String>) {
    var counter = 0
    val agent = Agent0()
    val set = MNIST.allSet()
    set.forEach { image ->
        val r = agent.recognize(image)
        if (image.index != r) {
            counter++
            image.save()
            println("${image.index} -> $r")
        } else image.delete()
    }
    println("${1.0 - counter*1.0/set.size}")
}

private operator fun List<Double>.plus(b: List<Double>) = zip(b).map { (a, b) -> a + b }
