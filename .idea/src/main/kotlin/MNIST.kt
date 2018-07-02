import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.*

object MNIST {
    private val mnistTrainPath = "G:\\ih8unem\\mnist_png\\training"
    private val mnistTestPath = "G:\\ih8unem\\mnist_png\\testing"
    const val errorPath = "error_images"

    fun buildBatch(size: Int, path: String = errorPath): Set<Image> {
        val dir = File(path)
        return (0 until size/10).flatMap {
            dir.listFiles().map {
                it.listFiles().toList().shuffled().first()
            }.map { Image(it) }
        }.toSet()
    }

    fun allSet(path: String = errorPath): Set<Image> {
        val dir = File(path)
        return dir.listFiles().flatMap {
            it.listFiles().map { Image(it) }
        }.toSet()
    }

    fun next(path: String = errorPath): Image {
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
    val agent = TestAgent()
    val set = MNIST.allSet()
    set.forEach { image ->
        try {
            val r = agent.recognize(image)
            if (image.index != r) throw RecognizeError()
            image.delete()
            println("${image.index} -> ok")
        } catch (e: RecognizeError) {
            counter++
            image.save()
            println("${image.index} -> fail")
        }
    }
    println("${1.0 - counter*1.0/set.size}")
}

private operator fun List<Double>.plus(b: List<Double>) = zip(b).map { (a, b) -> a + b }
