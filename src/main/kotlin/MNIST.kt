import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.*

object MNIST {
    val mnistTrainPath = "/home/melkor/mnist_png/training"
    val mnistTestPath = "/home/melkor/mnist_png/testing"
    const val errorPath = "error_images"

    fun buildBatch(size: Int, path: String = mnistTrainPath): Set<Image> {
        val dir = File(path)
        return (0 until size/10).flatMap { _ ->
            dir.listFiles().map {
                it.listFiles().toList().shuffled().first()
            }.map { Image(it) }
        }.toSet()
    }

    fun allSet(path: String = errorPath): Set<Image> {
        val dir = File(path)
        return dir.listFiles().flatMap { subdir ->
            subdir.listFiles().map { Image(it) }
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

// 60_000 -agent0-> 31551 -agent1-> 18097 -agent2-> 13393 -agent3-> 10346 -agent4-> 8529 -agent5-> 7452 -agent6-> 6366
fun main(args: Array<String>) {
    println(MNIST.allSet().size)
//    File(MNIST.errorPath).mkdir()
//    var counter = 0
//    val agent = TestAgent()
//    val set = MNIST.allSet()
//    set.forEach { image ->
//        try {
//            val r = agent.recognize(image)
//            if (image.index != r) throw RecognizeError()
//            image.delete()
//            println("${image.index} -> ok")
//        } catch (e: RecognizeError) {
//            counter++
//            image.save()
//            println("${image.index} -> fail")
//        }
//    }
//    println("${1.0 - counter*1.0/set.size}")
}

private operator fun List<Double>.plus(b: List<Double>) = zip(b).map { (a, b) -> a + b }
